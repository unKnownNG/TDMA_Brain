"""
sinr_auditor.py — SINR-aware physics auditor for TDMA schedules.

The core insight: G² (distance-2 coloring) assumes R_interference = R_decode.
In any real radio system this is physically false. Interference propagates
further than the decode boundary.

The interference-to-decode radius ratio for a given waveform and channel:

    R_interference / R_decode = 10^(SINR_req / (10 × α))

For SINR_req=10 dB, α=2 (free space) → ratio ≈ 3.16
G² silently sets this ratio to 1.0.

The three-node counterexample on a line:
    u = 0 m, v = 450 m, w = 970 m
    G: edge(u,v)=450m ✓, no edge(v,w)=520m ✗, no edge(u,w)=970m ✗
    G²: u and w have no shared neighbour → G² says they CAN share a slot.
    Assuming equal TX powers, free-space pathloss (α=2), and negligible
    thermal noise, the interference-limited SINR is approximately 1.26 dB.
    Under a hard-threshold model with SINR_req=10 dB, this transmission
    violates the configured reception criterion.

Model Assumptions (explicit):
    - Single shared frequency/channel.
    - One transmission assignment per node per frame.
    - Half-duplex radios.
    - Fixed, equal transmit power unless otherwise stated.
    - Euclidean topology with R=500 m for the graph model.
    - Synchronized slot boundaries.
    - Graph model treats 1-hop and 2-hop conflicts as forbidden
      simultaneous transmissions.
    - SINR model uses specified pathloss exponent, noise floor,
      antenna gains, and SINR threshold.
    - Fading/shadowing are excluded unless explicitly enabled in EMANE.
    - PCR (Packet Completion Rate) is hard-threshold: decode succeeds
      iff SINR ≥ threshold. EMANE can use SINR-dependent PCR curves;
      results may differ under soft-decision decoding.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .topology import Coordinates, build_connectivity_graph, euclidean_distance
from .conflicts import build_conflict_graph
from .solver import solve


# ---------------------------------------------------------------------------
# Radio parameter set (mirrors EMANE PHY configuration)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Explicit model assumptions (Review: must be inspectable and documentable)
# ---------------------------------------------------------------------------

MODEL_ASSUMPTIONS = [
    "Single shared frequency/channel.",
    "One transmission assignment per node per frame (cyclic TDMA).",
    "Half-duplex radios (cannot TX and RX simultaneously).",
    "Fixed, equal transmit power for all nodes unless otherwise stated.",
    "Euclidean 2D topology with configurable range R for the graph model.",
    "Synchronized slot boundaries (no clock drift modelled).",
    "Graph model: 1-hop and 2-hop (hidden-terminal) conflicts are forbidden"
    " simultaneous transmissions.",
    "SINR model: log-distance pathloss with configurable exponent α,"
    " thermal noise floor, isotropic antenna gains, and hard SINR threshold.",
    "Fading/shadowing excluded (deterministic propagation only).",
    "PCR model: hard-threshold (decode succeeds iff SINR ≥ threshold)."
    " EMANE supports SINR-dependent PCR curves; predictions may differ"
    " under soft-decision decoding.",
]


@dataclass
class RadioParams:
    """
    Physical-layer parameters matching EMANE's PHY configuration.
    All values must match what you configure in EMANE for predictions to hold.

    NOTE: This uses a hard-threshold decode model (SINR ≥ threshold → success).
    EMANE can use SINR-dependent Packet Completion Rate (PCR) curves, so a
    given SINR value does not universally mean 0% or 100% reception.
    """
    tx_power_dbm: float = 20.0          # Transmit power in dBm
    tx_antenna_gain_dbi: float = 0.0    # TX antenna gain (0 = isotropic)
    rx_antenna_gain_dbi: float = 0.0    # RX antenna gain (0 = isotropic)
    noise_figure_db: float = 6.0        # Receiver noise figure
    bandwidth_hz: float = 1_000_000.0   # Channel bandwidth (1 MHz default)
    sensitivity_dbm: float = -96.0      # Receiver sensitivity floor
    pathloss_exponent: float = 2.0      # α: 2=free-space, 3-4=urban/indoor
    sinr_threshold_db: float = 10.0     # Required SINR for successful decode
    reference_distance_m: float = 1.0   # Path-loss reference distance
    frequency_mhz: float = 2400.0       # Centre frequency in MHz

    @property
    def thermal_noise_dbm(self) -> float:
        """
        Thermal noise floor: N = -174 + 10·log10(BW) + NF  [dBm]
        At 1 MHz and NF=6 dB:  -174 + 60 + 6 = -108 dBm
        """
        return -174.0 + 10 * math.log10(self.bandwidth_hz) + self.noise_figure_db


# ---------------------------------------------------------------------------
# Per-link prediction
# ---------------------------------------------------------------------------

@dataclass
class LinkPrediction:
    tx_node: str
    rx_node: str
    slot: int
    rx_power_dbm: float               # Signal power at receiver
    interference_dbm: Optional[float] # Summed interference (None if no interferers above sensitivity)
    noise_floor_dbm: float            # Thermal noise floor
    sinr_db: float                    # Computed SINR
    above_sensitivity: bool           # Is signal above receiver sensitivity?
    passes_sinr: bool                 # Does signal meet SINR threshold?
    delivery_prediction: str          # "DELIVER" | "SINR_FAIL" | "BELOW_SENSITIVITY"
    active_interferers: List[str] = field(default_factory=list)  # Nodes causing interference
    g2_says_safe: bool = True         # Whether G² declared this link conflict-free


@dataclass
class SINRAuditReport:
    params: RadioParams
    link_predictions: List[LinkPrediction]
    g2_safe_but_sinr_fail: List[LinkPrediction]   # The interesting cases
    total_links: int
    predicted_deliveries: int
    predicted_failures: int
    interference_ratio: float                      # R_int / R_dec


# ---------------------------------------------------------------------------
# Physics functions
# ---------------------------------------------------------------------------

def free_space_pathloss_db(
    distance_m: float,
    pathloss_exponent: float = 2.0,
    reference_distance_m: float = 1.0,
    frequency_mhz: float = 2400.0,
) -> float:
    """
    Generalised path-loss model:

        L(d) = L(d0) + 10·α·log10(d / d0)

    where L(d0) is the free-space loss at the reference distance:

        L(d0) = 20·log10(4π·d0·f / c)   [Friis, d0=1m]

    For d0=1m, f=2.4 GHz:
        L(1m) = 20·log10(4π × 1 × 2.4e9 / 3e8) ≈ 40.05 dB

    Parameters
    ----------
    distance_m         : distance in metres (must be > 0)
    pathloss_exponent  : α (2 = free space, 3-4 = urban)
    reference_distance_m : d0 for the log-distance model
    frequency_mhz      : carrier frequency

    Returns
    -------
    Path loss in dB (positive value to subtract from TX power)
    """
    if distance_m <= 0:
        return 0.0
    if distance_m < reference_distance_m:
        distance_m = reference_distance_m

    # Free-space loss at reference distance (Friis)
    freq_hz = frequency_mhz * 1e6
    wavelength = 3e8 / freq_hz
    l_d0 = 20 * math.log10(4 * math.pi * reference_distance_m / wavelength)

    # Log-distance model
    pathloss = l_d0 + 10 * pathloss_exponent * math.log10(
        distance_m / reference_distance_m
    )
    return max(pathloss, 0.0)


def rx_power_dbm(
    tx_power_dbm: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    pathloss_db: float,
) -> float:
    """
    Received power (Friis link budget):
        Prx = Ptx + Gtx + Grx - L
    """
    return tx_power_dbm + tx_gain_dbi + rx_gain_dbi - pathloss_db


def sum_power_dbm(powers_dbm: List[float]) -> float:
    """
    Sum multiple power levels in dBm.

    Convert each from dBm to milliwatts, sum, convert back.
    This is how EMANE accumulates interference from multiple transmitters.
    """
    if not powers_dbm:
        return -float("inf")
    total_mw = sum(10 ** (p / 10) for p in powers_dbm)
    if total_mw <= 0:
        return -float("inf")
    return 10 * math.log10(total_mw)


def compute_sinr_db(
    signal_dbm: float,
    interference_powers_dbm: List[float],
    noise_dbm: float,
    sensitivity_dbm: float,
) -> Tuple[float, List[float]]:
    """
    Compute SINR following EMANE's noise model:

    1. Discard any interference source whose received power is below
       the sensitivity floor (EMANE discards sub-sensitivity energy
       before counting it as noise).
    2. Sum the remaining interference powers linearly.
    3. SINR = Signal / (Interference + Noise)

    Returns
    -------
    (sinr_db: float, active_interference_powers: List[float])
    """
    # EMANE rule: drop sub-sensitivity interference before summing
    active = [p for p in interference_powers_dbm if p >= sensitivity_dbm]

    if active:
        interference_total = sum_power_dbm(active)
        # Sum interference and noise
        noise_total = sum_power_dbm([interference_total, noise_dbm])
    else:
        noise_total = noise_dbm

    # SINR in linear
    signal_mw = 10 ** (signal_dbm / 10)
    noise_mw = 10 ** (noise_total / 10)

    if noise_mw <= 0:
        return float("inf"), active

    sinr_linear = signal_mw / noise_mw
    sinr_db = 10 * math.log10(sinr_linear)
    return sinr_db, active


# ---------------------------------------------------------------------------
# Interference radius insight
# ---------------------------------------------------------------------------

def interference_ratio(
    sinr_req_db: float, pathloss_exponent: float
) -> float:
    """
    Compute R_interference / R_decode.

    Derivation: at decode boundary d_c, signal equals (SINR_req × noise).
    At interference boundary d_i, signal equals noise.
    Since power ∝ d^(-α):
        (d_i / d_c)^α = SINR_req   →   d_i / d_c = SINR_req^(1/α)
        = 10^(SINR_req_dB / (10·α))

    For SINR=10 dB, α=2: ratio = 10^(10/20) = 10^0.5 ≈ 3.162
    G² assumes ratio = 1.0.
    """
    return 10 ** (sinr_req_db / (10 * pathloss_exponent))


# ---------------------------------------------------------------------------
# Main auditor
# ---------------------------------------------------------------------------

def _get_pos(G: nx.Graph, node: str) -> Tuple[float, float]:
    return G.nodes[node].get("pos", (0.0, 0.0))


def audit_schedule(
    G: nx.Graph,
    schedule: Dict[str, int],
    params: RadioParams,
    H: Optional[nx.Graph] = None,
) -> SINRAuditReport:
    """
    For every (tx, rx) pair where tx transmits in slot s and rx is a
    direct neighbour of tx in G (expected decode partner), compute SINR
    under all simultaneous transmitters in the same slot.

    EMANE's physical-layer rule (replicated here):
      1. Compute received power from signal tx at rx.
      2. Find all OTHER nodes transmitting in the same slot (co-slot interferers).
      3. Compute received power from each interferer at rx.
      4. Discard any interferer power below sensitivity_dbm.
      5. Sum remaining interference linearly (+ thermal noise).
      6. Compute SINR. Compare to sinr_threshold_db.

    Cases of interest: G² says a co-slot pair is safe, but SINR fails.
    These prove the graph model is insufficient.

    Parameters
    ----------
    G        : connectivity graph
    schedule : node → slot assignment
    params   : radio parameters (must match EMANE config)
    H        : conflict graph (optional; used to determine G²-safe pairs)
    """
    predictions: List[LinkPrediction] = []

    # Group nodes by slot
    slot_to_nodes: Dict[int, List[str]] = {}
    for node, slot in schedule.items():
        slot_to_nodes.setdefault(slot, []).append(node)

    noise_floor = params.thermal_noise_dbm

    for (tx, rx) in G.edges():
        # Only evaluate links where TX is actually transmitting
        slot = schedule.get(tx)
        if slot is None:
            continue

        # Co-slot transmitters (potential interferers at rx)
        co_slot_txs = [
            n for n in slot_to_nodes.get(slot, [])
            if n != tx
        ]

        # Signal: tx → rx
        pos_tx = _get_pos(G, tx)
        pos_rx = _get_pos(G, rx)
        d_signal = euclidean_distance(pos_tx, pos_rx)
        pl_signal = free_space_pathloss_db(
            d_signal, params.pathloss_exponent,
            params.reference_distance_m, params.frequency_mhz
        )
        p_signal = rx_power_dbm(
            params.tx_power_dbm,
            params.tx_antenna_gain_dbi,
            params.rx_antenna_gain_dbi,
            pl_signal,
        )
        above_sensitivity = p_signal >= params.sensitivity_dbm

        # Interference: each co-slot TX at rx
        interference_powers: List[float] = []
        interferer_names: List[str] = []
        for interferer in co_slot_txs:
            pos_int = _get_pos(G, interferer)
            d_int = euclidean_distance(pos_int, pos_rx)
            pl_int = free_space_pathloss_db(
                d_int, params.pathloss_exponent,
                params.reference_distance_m, params.frequency_mhz
            )
            p_int = rx_power_dbm(
                params.tx_power_dbm,
                params.tx_antenna_gain_dbi,
                params.rx_antenna_gain_dbi,
                pl_int,
            )
            interference_powers.append(p_int)
            interferer_names.append(interferer)

        sinr_db, active_interference = compute_sinr_db(
            p_signal, interference_powers, noise_floor, params.sensitivity_dbm
        )
        active_interferer_names = [
            interferer_names[i] for i, p in enumerate(interference_powers)
            if p >= params.sensitivity_dbm
        ]

        if not above_sensitivity:
            delivery = "BELOW_SENSITIVITY"
            passes = False
        elif sinr_db >= params.sinr_threshold_db:
            delivery = "DELIVER"
            passes = True
        else:
            delivery = "SINR_FAIL"
            passes = False

        int_dbm = sum_power_dbm(active_interference) if active_interference else None

        # Was this declared safe by G²?
        g2_safe = True
        if H is not None:
            # If any co-slot interferer conflicts with tx in H, G² caught it
            for interferer in co_slot_txs:
                if H.has_edge(tx, interferer):
                    g2_safe = False
                    break

        predictions.append(LinkPrediction(
            tx_node=tx,
            rx_node=rx,
            slot=slot,
            rx_power_dbm=round(p_signal, 2),
            interference_dbm=round(int_dbm, 2) if int_dbm is not None else None,
            noise_floor_dbm=round(noise_floor, 2),
            sinr_db=round(sinr_db, 2),
            above_sensitivity=above_sensitivity,
            passes_sinr=passes,
            delivery_prediction=delivery,
            active_interferers=active_interferer_names,
            g2_says_safe=g2_safe,
        ))

    # The money: G² says safe, physics says fail
    g2_safe_but_sinr_fail = [
        p for p in predictions
        if p.g2_says_safe and p.delivery_prediction != "DELIVER"
    ]

    ratio = interference_ratio(params.sinr_threshold_db, params.pathloss_exponent)

    return SINRAuditReport(
        params=params,
        link_predictions=predictions,
        g2_safe_but_sinr_fail=g2_safe_but_sinr_fail,
        total_links=len(predictions),
        predicted_deliveries=sum(1 for p in predictions if p.delivery_prediction == "DELIVER"),
        predicted_failures=sum(1 for p in predictions if p.delivery_prediction != "DELIVER"),
        interference_ratio=round(ratio, 3),
    )


# ---------------------------------------------------------------------------
# Price of physics curve
# ---------------------------------------------------------------------------

def price_of_physics_curve(
    coords: Coordinates,
    range_m: float = 500.0,
    sinr_values_db: Optional[List[float]] = None,
    pathloss_exponent: float = 2.0,
    timeout_sec: float = 2.0,
) -> List[Dict]:
    """
    Sweep SINR thresholds and compute the slot count needed by an
    SINR-aware conflict graph at each threshold.

    For each SINR_req:
      1. Compute R_int = R_dec × 10^(SINR_req / (10·α))
      2. u and w conflict if any of u's receivers is within R_int of w
         (the Gupta-Kumar protocol model / Physical Interference Model)
      3. Find optimal coloring of this SINR-aware conflict graph
      4. Record slot count

    Returns list of dicts:
        sinr_db, ratio, slot_count, lower_bound, optimality, vs_g2_baseline
    """
    if sinr_values_db is None:
        sinr_values_db = [0, 2, 4, 6, 8, 10, 12, 15, 20]

    # Baseline: G² (ratio=1, i.e., SINR_req=0 dB in the protocol model)
    G0 = build_connectivity_graph(coords, range_m)
    H0 = build_conflict_graph(G0)
    baseline_result = solve(H0, timeout_sec=timeout_sec)
    baseline_k = baseline_result.slot_count

    results = []
    for sinr_db in sinr_values_db:
        ratio = 10 ** (sinr_db / (10 * pathloss_exponent))
        # Build SINR-aware conflict graph:
        # u conflicts with w if there exists a receiver r (neighbour of u)
        # such that distance(w, r) ≤ range_m × ratio
        G_sinr = nx.Graph()
        nodes = list(coords.keys())
        for name, (x, y) in coords.items():
            pos_data = G0.nodes[name].get("pos", (float(x), float(y)))
            G_sinr.add_node(name, pos=pos_data)

        interference_range = range_m * ratio

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, w = nodes[i], nodes[j]
                # Check if u and w already conflict in G²
                if H0.has_edge(u, w):
                    G_sinr.add_edge(u, w)
                    continue
                # Check SINR-aware: does w interfere at any of u's receivers?
                conflict = False
                for r in G0.neighbors(u):
                    pos_w = coords[w]
                    pos_r = coords[r]
                    d_wr = euclidean_distance(pos_w, pos_r)
                    if d_wr <= interference_range:
                        conflict = True
                        break
                if not conflict:
                    # Also check: does u interfere at any of w's receivers?
                    for r in G0.neighbors(w):
                        pos_u = coords[u]
                        pos_r = coords[r]
                        d_ur = euclidean_distance(pos_u, pos_r)
                        if d_ur <= interference_range:
                            conflict = True
                            break
                if conflict:
                    G_sinr.add_edge(u, w)

        result = solve(G_sinr, timeout_sec=timeout_sec)
        results.append({
            "sinr_db": sinr_db,
            "ratio": round(ratio, 3),
            "slot_count": result.slot_count,
            "lower_bound": result.lower_bound,
            "optimality": result.optimality_status,
            "vs_g2_baseline": result.slot_count - baseline_k,
        })

    return results


def format_price_of_physics(rows: List[Dict], baseline_k: int) -> str:
    """Format the price-of-physics curve as a text table."""
    lines = [
        "",
        "PRICE OF PHYSICS — Slots needed vs. Required SINR",
        "=" * 65,
        f"  G² Baseline (ratio=1.0): {baseline_k} slots",
        "",
        f"{'SINR (dB)':>10}  {'R_int/R_dec':>12}  "
        f"{'Slots':>6}  {'LB':>4}  {'Status':>15}  {'vs G²':>6}",
        "-" * 65,
    ]
    for r in rows:
        diff = r["vs_g2_baseline"]
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        lines.append(
            f"{r['sinr_db']:>10.0f}  {r['ratio']:>12.3f}  "
            f"{r['slot_count']:>6}  {r['lower_bound']:>4}  "
            f"{r['optimality']:>15}  {diff_str:>6}"
        )
    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The killer counterexample
# ---------------------------------------------------------------------------

def run_line_counterexample(params: Optional[RadioParams] = None) -> Dict:
    """
    Three nodes on a line: u=0m, v=450m, w=970m.

    G²  says: u and w CAN share a slot (no conflict).
    Physics says: when u and w co-transmit, SINR at v (trying to receive from u)
                  is ≈ 1.26 dB — well below any 10 dB waveform threshold.

    Returns full calculation breakdown for documentation and slides.
    """
    if params is None:
        params = RadioParams()

    coords = {
        "u": (0.0, 0.0),
        "v": (450.0, 0.0),
        "w": (970.0, 0.0),
    }
    G = build_connectivity_graph(coords, range_m=500.0)
    H = build_conflict_graph(G)
    # G edges: u-v (450m). v-w is 520m (outside range). u-w is 970m.

    schedule = {"u": 0, "v": 1, "w": 0}  # G² says u,w can share slot 0

    # Manual physics calculation
    d_uv = 450.0   # signal path: u transmits, v receives
    d_wv = 520.0   # interference path: w transmits, v receives

    pl_uv = free_space_pathloss_db(d_uv, params.pathloss_exponent,
                                    params.reference_distance_m, params.frequency_mhz)
    pl_wv = free_space_pathloss_db(d_wv, params.pathloss_exponent,
                                    params.reference_distance_m, params.frequency_mhz)

    p_signal = rx_power_dbm(params.tx_power_dbm, params.tx_antenna_gain_dbi,
                              params.rx_antenna_gain_dbi, pl_uv)
    p_interference = rx_power_dbm(params.tx_power_dbm, params.tx_antenna_gain_dbi,
                                   params.rx_antenna_gain_dbi, pl_wv)

    # SINR at v (α=2 free space, simple ratio approach for insight)
    sinr_simple_db = 20 * params.pathloss_exponent * math.log10(d_wv / d_uv)

    sinr_db, _ = compute_sinr_db(
        p_signal, [p_interference], params.thermal_noise_dbm, params.sensitivity_dbm
    )

    g2_conflict = H.has_edge("u", "w")

    result = {
        "description": "Three nodes on a line: u=0m, v=450m, w=970m",
        "topology": {
            "u_v_distance_m": d_uv,
            "v_w_distance_m": d_wv,
            "u_w_distance_m": 970.0,
        },
        "g2_analysis": {
            "u_v_edge": G.has_edge("u", "v"),
            "v_w_edge": G.has_edge("v", "w"),
            "u_w_conflict": g2_conflict,
            "g2_verdict": "SAFE — u and w can share a slot" if not g2_conflict
                          else "CONFLICT — G² correctly caught this",
        },
        "physics_analysis": {
            "signal_path_m": d_uv,
            "interference_path_m": d_wv,
            "signal_pathloss_db": round(pl_uv, 2),
            "interference_pathloss_db": round(pl_wv, 2),
            "signal_power_at_v_dbm": round(p_signal, 2),
            "interference_power_at_v_dbm": round(p_interference, 2),
            "noise_floor_dbm": round(params.thermal_noise_dbm, 2),
            "sinr_simple_formula_db": round(sinr_simple_db, 2),
            "sinr_full_calculation_db": round(sinr_db, 2),
            "sinr_threshold_db": params.sinr_threshold_db,
            "result": "FAIL" if sinr_db < params.sinr_threshold_db else "PASS",
            "interference_ratio": round(
                interference_ratio(params.sinr_threshold_db, params.pathloss_exponent), 3
            ),
        },
        "conclusion": (
            f"G² says u and w can share a slot. "
            f"Assuming equal TX powers, free-space pathloss (α={params.pathloss_exponent}), "
            f"and negligible thermal noise, the interference-limited "
            f"SINR at v = {sinr_db:.2f} dB, which is well below the "
            f"{params.sinr_threshold_db} dB target SINR. Under the hard-threshold "
            f"model, this transmission violates the configured reception criterion. "
            f"G² is insufficient for this geometry."
        ) if sinr_db < params.sinr_threshold_db and not g2_conflict else (
            f"SINR at v = {sinr_db:.2f} dB ≥ {params.sinr_threshold_db} dB. "
            f"G² verdict: {'correct' if g2_conflict else 'happens to be right for this threshold'}."
        ),
    }
    return result


def format_counterexample_report(result: Dict) -> str:
    """Format the counterexample as a human-readable report block."""
    pa = result["physics_analysis"]
    ga = result["g2_analysis"]
    lines = [
        "",
        "THE G² COUNTEREXAMPLE — Three nodes on a line",
        "=" * 60,
        f"  u at 0m    v at 450m    w at 970m",
        "",
        "GRAPH ANALYSIS (G²):",
        f"  u↔v edge: {ga['u_v_edge']}  (450m ≤ 500m)",
        f"  v↔w edge: {ga['v_w_edge']}  (520m > 500m)",
        f"  u↔w conflict: {ga['u_w_conflict']}  (no shared neighbour)",
        f"  → G² verdict: {ga['g2_verdict']}",
        "",
        "PHYSICS (u transmits, w co-transmits same slot, v tries to receive from u):",
        f"  Signal power at v:        {pa['signal_power_at_v_dbm']:>8.2f} dBm  (path={pa['signal_path_m']}m)",
        f"  Interference power at v:  {pa['interference_power_at_v_dbm']:>8.2f} dBm  (path={pa['interference_path_m']}m)",
        f"  Thermal noise floor:      {pa['noise_floor_dbm']:>8.2f} dBm",
        f"  SINR at v:                {pa['sinr_full_calculation_db']:>8.2f} dB",
        f"  Required SINR:            {pa['sinr_threshold_db']:>8.2f} dB",
        f"  Result:                   {pa['result']}",
        "",
        f"  R_interference / R_decode = {pa['interference_ratio']:.3f}×",
        f"  (G² assumes this ratio = 1.0)",
        "",
        f"CONCLUSION: {result['conclusion']}",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_audit_report(audit: SINRAuditReport) -> str:
    """Format the full SINR audit report as readable text."""
    p = audit.params
    lines = [
        "",
        "SINR AUDIT REPORT",
        "=" * 65,
        f"Radio Parameters:",
        f"  TX power:         {p.tx_power_dbm} dBm",
        f"  Pathloss exponent: α = {p.pathloss_exponent}",
        f"  SINR threshold:   {p.sinr_threshold_db} dB",
        f"  Noise figure:     {p.noise_figure_db} dB",
        f"  Sensitivity:      {p.sensitivity_dbm} dBm",
        f"  Noise floor:      {p.thermal_noise_dbm:.1f} dBm",
        "",
        f"Interference Radius Ratio: R_int / R_dec = {audit.interference_ratio:.3f}×",
        f"(G² assumes this = 1.0)",
        "",
        f"Results:",
        f"  Total evaluated links:   {audit.total_links}",
        f"  Predicted deliveries:    {audit.predicted_deliveries}",
        f"  Predicted failures:      {audit.predicted_failures}",
        f"  G²-safe but SINR-fail:   {len(audit.g2_safe_but_sinr_fail)}  ← G² model failures",
    ]

    if audit.g2_safe_but_sinr_fail:
        lines.append("")
        lines.append("⚠ LINKS G² SAYS SAFE BUT PHYSICS FAILS:")
        lines.append("-" * 65)
        for lp in audit.g2_safe_but_sinr_fail:
            lines.append(
                f"  Slot {lp.slot}: {lp.tx_node}→{lp.rx_node}  "
                f"SINR={lp.sinr_db:.1f}dB < {p.sinr_threshold_db}dB  "
                f"[{lp.delivery_prediction}]"
            )
            if lp.active_interferers:
                lines.append(f"         Interferers: {lp.active_interferers}")
    else:
        lines.append("")
        lines.append("✓ No G²-safe links fail under physics for this topology.")
        lines.append("  (May be topology-dependent. Try denser or different layouts.)")

    lines.append("=" * 65)
    return "\n".join(lines)
