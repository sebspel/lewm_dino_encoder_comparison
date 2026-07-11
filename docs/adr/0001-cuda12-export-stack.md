# ADR 0001 — CUDA-12 export stack & calibration EP split

**Status:** Accepted · **Date:** 2026-07-10

## Context

The export/quantization stack (TensorRT + NVIDIA TensorRT Model Optimizer + its
ONNX Runtime dependency) must stay binary-compatible with the pod's CUDA 12.4 driver
and the uv-locked `cu124` torch. Several dependencies now default to CUDA-13 wheels,
which pull `nvidia-*-cu13` and fail to initialize cuDNN against a 12.x driver.

## Decision

- **TensorRT** is installed by `setup.sh` (cu12, CUDA-12.4-matched) and kept **out of uv**
  so it cannot pull a conflicting `libnvinfer`/CUDA stack. Do not pin `tensorrt` in uv.
- **NVIDIA TensorRT Model Optimizer** (`nvidia-modelopt[onnx]`, the explicit-INT8 Q/DQ tool)
  is installed the same way — by `setup.sh`, out of uv — so its ONNX Runtime / CUDA stack
  stays matched to CUDA 12.4.
- Two CUDA-13 vectors are pinned out:
  1. `onnxruntime-gpu` is installed from onnxruntime's dedicated **CUDA-12 feed** (the default
     PyPI wheel is CUDA-13). Installing it **before** modelopt prevents modelopt's unbounded
     dependency from re-resolving it to the cu13 default. This CUDA-12 build is also what lets
     the Model-Optimizer calibration pass run on the GPU (CUDA EP).
  2. `nvidia-modelopt` is pinned to a build compatible with the locked cu124 torch; its latest
     requires a CUDA-13 torch (2.13). `setup.sh` pins torch to the installed cu124 build so an
     upgrade fails loudly rather than silently swapping the CUDA stack.
- Confirmed torch-2.6-compatible pins: `modelopt==0.43.0`, `onnxruntime-gpu==1.24.4` (cu12).
  modelopt 0.43.0 also caps `onnx==1.19.1` vs the uv-locked 1.22.0 — **lock harmonization is
  open (owner)**.

## Calibration execution-provider (EP) split

The encoder calibrates on the **GPU (CUDA EP)**; the predictor calibrates on the **CPU EP**.

The `onnxruntime-gpu` CUDA EP miscomputes the predictor's dynamic-batch reshape
(`Squeeze(Shape(latent))` → head-split `Reshape`), fabricating a target of 192 (= 8×8×3) at
batch 8 and crashing modelopt's MHA-exclusion probe. The CPU EP (and native TensorRT) computes
it correctly. **EP choice affects calibration speed only, not the derived per-tensor scales**,
so the split is plumbing, not a result-affecting decision.

## Consequences

- The whole export stack stays on **CUDA major 12**.
- FP32/FP16 build data-free from the base ONNX; **INT8 is explicit Q/DQ** — modelopt inserts
  Q/DQ and bakes per-tensor scales from a calibration pass, and TensorRT honors the embedded
  Q/DQ instead of calibrating at build time.
- Fallback if the FP16 cast of the non-quantized remainder drifts unacceptably: keep the
  remainder FP32 via modelopt's `high_precision_dtype` (see SPEC, "INT8 means INT8+FP16").
