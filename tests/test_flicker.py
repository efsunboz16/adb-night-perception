import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from eval_video import short_runs  # noqa: E402

D, L = True, False   # kısık, açık


def test_brief_opening_is_flicker():
    assert short_runs([[D], [D], [L], [D], [D]], 5) == 1


def test_single_permanent_change_is_not_flicker():
    assert short_runs([[D]] * 3 + [[L]] * 10, 5) == 0


def test_brief_dimming_counted_long_state_not():
    assert short_runs([[L], [D], [D], [L], [L], [L], [L], [L], [L], [D]], 5) == 1