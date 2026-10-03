"""
cli.py — Entry point for the TDMA Brain.

Usage:
  python -m tdma_brain '{"Node_01":[0,0], ...}'
  python -m tdma_brain --file coords.json
  cat coords.json | python -m tdma_brain --stdin

All flags are documented in README.md.
"""

import argparse
import json
import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import Dict, Tuple, Optional

from .topology import build_connectivity_graph, get_node_order, topology_stats, Coordinates
from .conflicts import build_conflict_graph, conflict_stats
from .validator import validate_schedule, validate_matrix
from .reporter import (
    build_matrix,
    format_ascii_report,
    format_benchmark_table,
    build_json_output,
    generate_html_topology,
)
from .analyzer import (
    robustness_sweep,
    format_robustness_report,
    format_all_explanations,
)
from .sinr_auditor import (
    RadioParams,
    MODEL_ASSUMPTIONS,
    audit_schedule,
    format_audit_report,
    run_line_counterexample,
    format_counterexample_report,
    price_of_physics_curve,
    format_price_of_physics,
)
from .sinr_repair import (
    repair_schedule_sinr,
    format_repair_report,
)
from .solver import (
    solve,
    benchmark_runtime_statistics,
    format_runtime_statistics,
)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def parse_coords(raw: str) -> Coordinates:
    """
    Parse a JSON string into a Coordinates dict.
    Validates: dict, ≥1 node, each value is [x, y].
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Input must be a JSON object (dict).")
    if not data:
        raise ValueError("Input has no nodes.")

    coords: Coordinates = {}
    for name, val in data.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"Node name must be a non-empty string, got: {name!r}")
        if not isinstance(val, (list, tuple)) or len(val) != 2:
            raise ValueError(f"Node {name!r}: value must be [x, y], got: {val!r}")
        try:
            x, y = float(val[0]), float(val[1])
        except (TypeError, ValueError) as e:
            raise ValueError(f"Node {name!r}: coordinates must be numeric: {e}") from e
        import math
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Node {name!r}: coordinates must be finite numbers.")
        coords[name] = (x, y)

    return coords


def load_coords(args: argparse.Namespace) -> Coordinates:
    """Load coordinates from CLI argument, file, or stdin."""
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            raw = f.read()
    elif args.stdin:
        raw = sys.stdin.read()
    elif args.coords:
        raw = args.coords
    else:
        raise ValueError(
            "No input provided. Use a JSON string, --file, or --stdin."
        )
    return parse_coords(raw)


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tdma_brain",
        description="Certified-optimal TDMA scheduler under the G² conflict model, "
                    "with physical-layer SINR auditing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "coords", nargs="?",
        help='JSON string: \'{"Node_01":[x,y], ...}\''
    )
    p.add_argument("--file", "-f", metavar="PATH",
                   help="Read coordinates from a JSON file.")
    p.add_argument("--stdin", action="store_true",
                   help="Read coordinates from stdin.")
    p.add_argument("--range", "-r", type=float, default=500.0, metavar="METERS",
                   help="Communication range in metres (default: 500).")
    p.add_argument("--seed", "-s", type=int, default=None, metavar="INT",
                   help="Random seed for reproducibility.")
    p.add_argument("--sinr-model", action="store_true",
                   help="Enable SINR-aware physics audit (Part 2 innovation).")
    p.add_argument("--sinr-threshold", type=float, default=10.0, metavar="DB",
                   help="Required SINR in dB (default: 10).")
    p.add_argument("--pathloss-exp", type=float, default=2.0, metavar="FLOAT",
                   help="Pathloss exponent α (default: 2.0, free space).")
    p.add_argument("--counterexample", action="store_true",
                   help="Run the three-node line counterexample.")
    p.add_argument("--price-of-physics", action="store_true",
                   help="Compute K vs. SINR 'price of physics' curve.")
    p.add_argument("--robustness", action="store_true",
                   help="Run δ-bounded movement robustness sweep.")
    p.add_argument("--explain", action="store_true",
                   help="Show per-node conflict explanations.")
    p.add_argument("--benchmark", action="store_true",
                   help="Run heuristic portfolio comparison.")
    p.add_argument("--sinr-repair", action="store_true",
                   help="Run SINR-aware repair loop: augment conflict graph "
                        "with physics-discovered conflicts and re-color.")
    p.add_argument("--runtime-bench", action="store_true",
                   help="Run solver runtime benchmark over random topologies "
                        "(reports median, P95, max).")
    p.add_argument("--assumptions", action="store_true",
                   help="Print explicit model assumptions and exit.")
    p.add_argument("--html", metavar="PATH",
                   help="Write SVG topology visualisation to HTML file.")
    p.add_argument("--json", metavar="PATH",
                   help="Write full JSON result to file.")
    p.add_argument("--verbose", action="store_true",
                   help="Print detailed graph statistics.")
    return p


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Special: print assumptions and exit
    if args.assumptions:
        print("\nMODEL ASSUMPTIONS")
        print("=" * 60)
        for i, assumption in enumerate(MODEL_ASSUMPTIONS, 1):
            print(f"  {i:>2}. {assumption}")
        print("=" * 60)
        return 0

    # Special: runtime benchmark (no coords needed)
    if args.runtime_bench:
        print("Running solver runtime benchmark (100 random 16-node topologies)...")
        stats = benchmark_runtime_statistics(
            num_topologies=100,
            node_count=16,
            range_m=args.range,
        )
        print(format_runtime_statistics(stats))
        return 0

    # Special: standalone counterexample (no coords needed)
    if args.counterexample and not (args.coords or args.file or args.stdin):
        params = RadioParams(
            sinr_threshold_db=args.sinr_threshold,
            pathloss_exponent=args.pathloss_exp,
        )
        result = run_line_counterexample(params)
        print(format_counterexample_report(result))
        return 0

    # Load coordinates
    try:
        coords = load_coords(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    range_m = args.range
    node_order = get_node_order(coords)

    # Step 1: Build graphs
    G = build_connectivity_graph(coords, range_m)
    H = build_conflict_graph(G)

    if args.verbose:
        topo = topology_stats(G)
        conf = conflict_stats(H)
        print(f"[topology] {topo}")
        print(f"[conflicts] {conf}")

    # Step 2: Solve
    result = solve(
        H,
        seed=args.seed,
        run_benchmark=args.benchmark,
    )

    # Step 3: Validate
    validation = validate_schedule(G, result.node_to_slot, node_order)
    if not validation.is_valid:
        print("CRITICAL: Solver produced an invalid schedule!", file=sys.stderr)
        for v in validation.violations:
            print(f"  {v}", file=sys.stderr)
        # Still report, but exit with error
        return_code = 2
    else:
        return_code = 0

    # Also validate matrix
    matrix = build_matrix(result.node_to_slot, node_order)
    matrix_val = validate_matrix(matrix, result.node_to_slot, node_order)
    if not matrix_val.is_valid:
        print("CRITICAL: Matrix validation failed!", file=sys.stderr)
        for v in matrix_val.violations:
            print(f"  {v}", file=sys.stderr)
        return_code = 2

    # Step 4: ASCII report (always printed to stdout)
    print(format_ascii_report(result, G, node_order, range_m, validation))

    # Step 5: Optional outputs
    if args.benchmark and result.benchmark:
        print(format_benchmark_table(result.benchmark))

    if args.explain:
        print(format_all_explanations(G, H, result.node_to_slot, node_order))

    if args.robustness:
        points = robustness_sweep(coords, range_m)
        print(format_robustness_report(points))

    if args.sinr_model:
        params = RadioParams(
            sinr_threshold_db=args.sinr_threshold,
            pathloss_exponent=args.pathloss_exp,
        )
        audit = audit_schedule(G, result.node_to_slot, params, H)
        print(format_audit_report(audit))

        if args.counterexample:
            ce = run_line_counterexample(params)
            print(format_counterexample_report(ce))

        if args.price_of_physics:
            pop = price_of_physics_curve(
                coords, range_m,
                pathloss_exponent=args.pathloss_exp,
            )
            baseline_k = result.slot_count
            print(format_price_of_physics(pop, baseline_k))

    if args.sinr_repair:
        params = RadioParams(
            sinr_threshold_db=args.sinr_threshold,
            pathloss_exponent=args.pathloss_exp,
        )
        repair_result = repair_schedule_sinr(
            coords, range_m, params,
            seed=args.seed,
        )
        print(format_repair_report(repair_result))

    # Step 6: File outputs
    if args.html:
        html = generate_html_topology(
            G, result.node_to_slot, coords,
            clique_witness=result.clique_witness,
        )
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[✓] Topology map saved to {args.html}")

    if args.json:
        output = build_json_output(
            result, G, node_order, coords, range_m, validation
        )
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"[✓] Full results saved to {args.json}")

    return return_code


if __name__ == "__main__":
    sys.exit(main())
