"""
test_sinr_auditor.py — Tests for sinr_auditor.py

Covers the counterexample, interference ratio formula, physics functions,
and the auditor's ability to find G²-safe-but-SINR-fail cases.
"""

import math
import pytest

from tdma_brain.sinr_auditor import (
    RadioParams,
    free_space_pathloss_db,
    rx_power_dbm,
    sum_power_dbm,
    compute_sinr_db,
    interference_ratio,
    run_line_counterexample,
    audit_schedule,
)
from tdma_brain.topology import build_connectivity_graph
from tdma_brain.conflicts import build_conflict_graph


# ---------------------------------------------------------------------------
# interference_ratio formula
# ---------------------------------------------------------------------------

def test_interference_ratio_free_space_10db():
    """For SINR=10 dB, α=2: ratio = 10^(10/20) = √10 ≈ 3.162."""
    ratio = interference_ratio(10.0, 2.0)
    assert ratio == pytest.approx(10 ** 0.5, rel=1e-4)

def test_interference_ratio_0db():
    """SINR=0 dB → ratio = 1.0 (this is what G² assumes)."""
    ratio = interference_ratio(0.0, 2.0)
    assert ratio == pytest.approx(1.0, rel=1e-6)

def test_interference_ratio_urban_alpha4():
    """For SINR=10 dB, α=4: ratio = 10^(10/40) = 10^0.25 ≈ 1.778."""
    ratio = interference_ratio(10.0, 4.0)
    assert ratio == pytest.approx(10 ** 0.25, rel=1e-4)


# ---------------------------------------------------------------------------
# free_space_pathloss_db
# ---------------------------------------------------------------------------

def test_pathloss_increases_with_distance():
    """Further distance → higher pathloss."""
    pl_100 = free_space_pathloss_db(100.0)
    pl_500 = free_space_pathloss_db(500.0)
    assert pl_500 > pl_100

def test_pathloss_reference_distance():
    """At reference distance (1m), pathloss equals the Friis constant."""
    pl = free_space_pathloss_db(1.0, pathloss_exponent=2.0, frequency_mhz=2400.0)
    # Friis at 1m, 2.4 GHz: ≈ 40 dB
    assert 35.0 < pl < 45.0

def test_pathloss_zero_distance_handled():
    """d ≤ 0 should not raise, returns 0."""
    pl = free_space_pathloss_db(0.0)
    assert pl == 0.0


# ---------------------------------------------------------------------------
# sum_power_dbm
# ---------------------------------------------------------------------------

def test_sum_power_two_equal():
    """Two equal powers: sum = power + 3dB."""
    result = sum_power_dbm([-50.0, -50.0])
    assert result == pytest.approx(-47.0, abs=0.1)

def test_sum_power_empty():
    result = sum_power_dbm([])
    assert result == -float("inf")

def test_sum_power_single():
    assert sum_power_dbm([-60.0]) == pytest.approx(-60.0, abs=0.01)


# ---------------------------------------------------------------------------
# compute_sinr_db — EMANE's sensitivity-floor rule
# ---------------------------------------------------------------------------

def test_sub_sensitivity_interference_discarded():
    """
    EMANE rule: interference below sensitivity floor is NOT counted as noise.
    Signal = -70 dBm, sensitivity = -90 dBm, interference = -100 dBm (below sensitivity).
    Interference should be discarded; SINR = signal / noise_only.
    """
    params = RadioParams(sensitivity_dbm=-90.0)
    sinr, active = compute_sinr_db(
        signal_dbm=-70.0,
        interference_powers_dbm=[-100.0],   # below sensitivity
        noise_dbm=params.thermal_noise_dbm,
        sensitivity_dbm=-90.0,
    )
    assert active == []  # Discarded
    # SINR should be high since no effective interference
    assert sinr > 10.0

def test_above_sensitivity_interference_counted():
    """Interference above sensitivity IS counted."""
    params = RadioParams(sensitivity_dbm=-90.0)
    sinr_with, active = compute_sinr_db(
        signal_dbm=-70.0,
        interference_powers_dbm=[-80.0],    # above sensitivity
        noise_dbm=params.thermal_noise_dbm,
        sensitivity_dbm=-90.0,
    )
    assert active == [-80.0]
    sinr_without, _ = compute_sinr_db(
        signal_dbm=-70.0,
        interference_powers_dbm=[],
        noise_dbm=params.thermal_noise_dbm,
        sensitivity_dbm=-90.0,
    )
    assert sinr_with < sinr_without  # Interference reduces SINR


# ---------------------------------------------------------------------------
# The three-node counterexample
# ---------------------------------------------------------------------------

def test_counterexample_g2_says_safe():
    """G² must NOT flag u and w as conflicting (they share no neighbour)."""
    result = run_line_counterexample()
    assert result["g2_analysis"]["u_w_conflict"] is False
    assert "SAFE" in result["g2_analysis"]["g2_verdict"]

def test_counterexample_sinr_fails():
    """
    Physics: with a 10 dB SINR threshold, u and w co-transmitting
    causes SINR < threshold at v (receiver of u).
    """
    params = RadioParams(sinr_threshold_db=10.0, pathloss_exponent=2.0)
    result = run_line_counterexample(params)
    sinr = result["physics_analysis"]["sinr_full_calculation_db"]
    assert sinr < params.sinr_threshold_db, (
        f"Expected SINR < {params.sinr_threshold_db} dB at v, got {sinr:.2f} dB"
    )
    assert result["physics_analysis"]["result"] == "FAIL"

def test_counterexample_simple_formula():
    """
    Simple SINR formula using the log-distance model with pathloss exponent α.
    The code computes: sinr_simple = 20·α·log10(d_int / d_sig)
    For default α=2.0, d_int=520, d_sig=450:
      SINR = 20 * 2 * log10(520/450) ≈ 2.51 dB
    Still below the 10 dB threshold — counterexample holds.
    """
    result = run_line_counterexample()
    sinr_simple = result["physics_analysis"]["sinr_simple_formula_db"]
    params = RadioParams()  # default α=2.0
    expected = 20 * params.pathloss_exponent * math.log10(520 / 450)
    assert sinr_simple == pytest.approx(expected, abs=0.1)
    # Key: still below the 10 dB threshold regardless of formula variant
    assert sinr_simple < params.sinr_threshold_db

def test_counterexample_ratio():
    """Interference ratio must be ≈ 3.162 for 10 dB threshold, α=2."""
    result = run_line_counterexample()
    ratio = result["physics_analysis"]["interference_ratio"]
    assert ratio == pytest.approx(10 ** 0.5, rel=0.01)


# ---------------------------------------------------------------------------
# audit_schedule — no interferers → all links deliver
# ---------------------------------------------------------------------------

def test_audit_isolated_pair_delivers():
    """
    Two isolated nodes — no co-slot interferers → signal delivers cleanly.
    """
    coords = {"A": (0.0, 0.0), "B": (400.0, 0.0)}
    G = build_connectivity_graph(coords)
    schedule = {"A": 0, "B": 1}   # Different slots, no co-transmission
    params = RadioParams()
    audit = audit_schedule(G, schedule, params)
    assert all(p.delivery_prediction == "DELIVER" for p in audit.link_predictions)


# ---------------------------------------------------------------------------
# RadioParams — thermal noise floor formula
# ---------------------------------------------------------------------------

def test_thermal_noise_1mhz():
    """At 1 MHz bandwidth, NF=6 dB: noise = -174 + 60 + 6 = -108 dBm."""
    params = RadioParams(bandwidth_hz=1_000_000.0, noise_figure_db=6.0)
    assert params.thermal_noise_dbm == pytest.approx(-108.0, abs=0.01)
