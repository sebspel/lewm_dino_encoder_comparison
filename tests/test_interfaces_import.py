"""Phase 0 smoke: the owned typed contract imports cleanly in-container.

Real adapter/export/benchmark tests arrive in Phase 4.
"""


def test_interfaces_imports():
    from src import interfaces

    assert interfaces.ExportConfig().action_dim == 2
