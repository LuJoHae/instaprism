from __future__ import annotations

from instaprism.__main__ import main


def test_main_returns_zero() -> None:
    assert main() == 0
