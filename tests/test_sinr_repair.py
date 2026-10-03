"""
test_sinr_repair.py — Tests for sinr_repair.py

Covers the SINR-aware schedule repair loop, verifying that:
  1. The repair loop converges.
  2. The repaired schedule uses ≥ the G² slot count.
  3. The 3-node counterexample is fixed by repair.
"""

import pytest

from tdma_brain.sinr_repair import repair_schedule_sinr, SINRRepairResult
from tdma_brain.sinr_auditor import RadioParams


# ---------------------------------------------------------------------------
# 3-node line counterexample: the repair loop must fix it
# ---------------------------------------------------------------------------

def test_repair_counterexample_converges():
    """Repair loop on the 3-node line must converge (no infinite loop)."""
    coords = {
        "u": (0.0, 0.0),
        "v": (450.0, 0.0),
        "w": (970.0, 0.0),
    }
    params = RadioParams(sinr_threshold_db=10.0, pathloss_exponent=2.0)
    result = repair_schedule_sinr(coords, range_m=500.0, params=params)

    assert isinstance(result, SINRRepairResult)
    assert len(result.iterations) <= 10


def test_repair_counterexample_separates_interferers():
    """
    The G² schedule puts u and w in the same slot.
    After repair, u and w must be in DIFFERENT slots.
    (The repair may or may not need a 3rd slot — it can move w to v's slot
    since v and w are not G-neighbors.)
    """
    coords = {
        "u": (0.0, 0.0),
        "v": (450.0, 0.0),
        "w": (970.0, 0.0),
    }
    params = RadioParams(sinr_threshold_db=10.0, pathloss_exponent=2.0)
    result = repair_schedule_sinr(coords, range_m=500.0, params=params)

    assert result.g2_slot_count == 2, f"Expected G² = 2 slots, got {result.g2_slot_count}"
    # u and w must now be in different slots
    assert result.repaired_schedule["u"] != result.repaired_schedule["w"], (
        f"u and w should be separated: u={result.repaired_schedule['u']}, "
        f"w={result.repaired_schedule['w']}"
    )
    assert result.repaired_slot_count >= result.g2_slot_count


def test_repair_counterexample_resolves_failures():
    """After repair, there should be no remaining SINR failures."""
    coords = {
        "u": (0.0, 0.0),
        "v": (450.0, 0.0),
        "w": (970.0, 0.0),
    }
    params = RadioParams(sinr_threshold_db=10.0, pathloss_exponent=2.0)
    result = repair_schedule_sinr(coords, range_m=500.0, params=params)

    assert result.remaining_sinr_failures == 0, (
        f"Expected 0 remaining failures, got {result.remaining_sinr_failures}"
    )


def test_repair_counterexample_edges_added():
    """The repair must add physics-discovered conflict edges."""
    coords = {
        "u": (0.0, 0.0),
        "v": (450.0, 0.0),
        "w": (970.0, 0.0),
    }
    params = RadioParams(sinr_threshold_db=10.0, pathloss_exponent=2.0)
    result = repair_schedule_sinr(coords, range_m=500.0, params=params)

    assert result.total_edges_added > 0, "Expected at least one physics edge added"


# ---------------------------------------------------------------------------
# Grid topology: no repair needed if no SINR failures
# ---------------------------------------------------------------------------

def test_repair_grid_400m_no_change_needed():
    """
    For the 400m grid with default radio params, check that repair
    completes and the slot count is ≥ the G² count.
    """
    coords = {
        f"Node_{r*4+c+1:02d}": (c * 400.0, r * 400.0)
        for r in range(4) for c in range(4)
    }
    params = RadioParams(sinr_threshold_db=10.0, pathloss_exponent=2.0)
    result = repair_schedule_sinr(coords, range_m=500.0, params=params)

    assert result.repaired_slot_count >= result.g2_slot_count
    assert result.g2_slot_count == 5


# ---------------------------------------------------------------------------
# Isolated pair: no repair needed
# ---------------------------------------------------------------------------

def test_repair_isolated_pair_no_change():
    """Two isolated nodes in different slots — no repair needed."""
    coords = {"A": (0.0, 0.0), "B": (400.0, 0.0)}
    params = RadioParams()
    result = repair_schedule_sinr(coords, range_m=500.0, params=params)

    assert result.price_of_physics == 0
    assert result.remaining_sinr_failures == 0
    assert result.total_edges_added == 0
