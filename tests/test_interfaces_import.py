"""Phase 0 smoke: the owned typed contract imports cleanly in-container.

Real adapter/export/benchmark tests arrive in Phase 4.
"""


def test_interfaces_imports():
    from src import interfaces

    # action_dim is the model-facing action width (frameskip pack), not the env ACTION_DIM.
    assert interfaces.ExportConfig().action_dim == interfaces.MODEL_ACTION_DIM == 10
    assert interfaces.ExportConfig().proprio_dim == interfaces.DINO_PROPRIO_DIM == 4
