from src.env import load_env

# Read `.env` once, before any module resolves a path or opens a W&B run.
load_env()
