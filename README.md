# TDMA Brain — Vaan Megam Networks

A certified-optimal TDMA scheduler under the Distance-2 (G²) conflict model,
with physical-layer SINR auditing for RF interference beyond the graph model.

## Three Claims

1. **Mathematical Guarantee** — Exact minimum-slot TDMA scheduling under a
   distance-2 conflict model, with a lower-bound certificate (maximum clique)
   and independent 6-invariant verification.
2. **Physical-Layer Challenge** — Demonstrates that graph-distance scheduling
   is not sufficient to guarantee RF-level interference safety (the G²
   counterexample).
3. **Emulator Validation** — Bridges schedules and continuous pairwise pathloss
   into EMANE and validates behaviour using receiver, SINR, and packet-delivery
   measurements.

## Quick Start

```bash
pip install -r requirements.txt

# Basic usage — JSON string
python -m tdma_brain '{"Node_01":[0,0],"Node_02":[400,0],"Node_03":[800,0],"Node_04":[0,400],"Node_05":[400,400],"Node_06":[800,400],"Node_07":[0,800],"Node_08":[400,800],"Node_09":[800,800],"Node_10":[1200,0],"Node_11":[1200,400],"Node_12":[1200,800],"Node_13":[0,1200],"Node_14":[400,1200],"Node_15":[800,1200],"Node_16":[1200,1200]}'

# From file
python -m tdma_brain --file examples/grid_4x4_400m.json

# Full analysis
python -m tdma_brain --file examples/grid_4x4_400m.json --sinr-model --robustness --explain --benchmark --html out.html --json out.json

# SINR repair loop (two-layer planner)
python -m tdma_brain --file examples/grid_4x4_400m.json --sinr-repair

# Print model assumptions
python -m tdma_brain --assumptions

# Runtime benchmark (reports median, P95, max over 100 topologies)
python -m tdma_brain --runtime-bench
```

## Model Assumptions

> These are explicitly documented in the code and can be printed with `--assumptions`.

1. Single shared frequency/channel.
2. One transmission assignment per node per frame (cyclic TDMA).
3. Half-duplex radios (cannot TX and RX simultaneously).
4. Fixed, equal transmit power for all nodes unless otherwise stated.
5. Euclidean 2D topology with configurable range R for the graph model.
6. Synchronized slot boundaries (no clock drift modelled).
7. Graph model: 1-hop and 2-hop (hidden-terminal) conflicts are forbidden simultaneous transmissions.
8. SINR model: log-distance pathloss with configurable exponent α, thermal noise floor, isotropic antenna gains, and hard SINR threshold.
9. Fading/shadowing excluded (deterministic propagation only).
10. PCR model: hard-threshold decode (SINR ≥ threshold → success). EMANE supports SINR-dependent PCR curves; predictions may differ under soft-decision decoding.

## Complexity & Scalability

| Phase | Complexity | Notes |
|:---|:---|:---|
| Graph construction G | O(N²) | Pairwise geometric connectivity |
| Conflict graph G² | O(N·Δ²) | Δ = max degree in G |
| Max clique ω(H) | Exponential | Practical for N=16 |
| Exact coloring χ(H) | NP-hard | B&B with DSATUR; practical for N=16 |
| SINR audit | O(N·K·N) | Per-link SINR under all co-slot transmitters |
| SINR repair loop | O(I·solve) | I = iterations until convergence |

Exact optimisation is practical for the specified 16-node problem size while
providing a proof of optimality that heuristic-only schedulers cannot. Use
`--runtime-bench` for rigorous measurements with median, P95, and maximum.

## Flags

| Flag | Description |
|:---|:---|
| `--file PATH` | Read coordinates from JSON file |
| `--stdin` | Read coordinates from stdin |
| `--range METERS` | Communication range (default: 500) |
| `--seed INT` | Random seed for reproducibility |
| `--sinr-model` | Enable SINR-aware conflict analysis |
| `--sinr-threshold DB` | Required SINR in dB (default: 10) |
| `--pathloss-exp FLOAT` | Pathloss exponent α (default: 2.0) |
| `--sinr-repair` | Run SINR-aware repair loop (two-layer planner) |
| `--counterexample` | Run the three-node G² vs physics counterexample |
| `--price-of-physics` | Compute K vs SINR tradeoff curve |
| `--robustness` | Run δ-robustness sweep |
| `--explain` | Show per-node conflict explanations |
| `--benchmark` | Run heuristic comparison |
| `--runtime-bench` | Run solver runtime statistics (median/P95/max) |
| `--assumptions` | Print model assumptions and exit |
| `--html PATH` | Generate topology visualisation |
| `--json PATH` | Save full results as JSON |

## Architecture

```
G → G² → Exact Coloring → Candidate Schedule
                                  ↓
                           SINR Audit
                                  ↓
                     Find Physical Conflicts
                                  ↓
                    Augment Conflict Graph
                                  ↓
                           Re-color
                                  ↓
                     EMANE Validation
```

## Key Files

| File | Purpose |
|:---|:---|
| [topology.py](tdma_brain/topology.py) | Connectivity graph with squared-distance boundary |
| [conflicts.py](tdma_brain/conflicts.py) | G² conflict graph via `nx.power(G, 2)` |
| [solver.py](tdma_brain/solver.py) | Exact branch-and-bound + heuristic portfolio |
| [validator.py](tdma_brain/validator.py) | 6-invariant independent validation |
| [sinr_auditor.py](tdma_brain/sinr_auditor.py) | SINR physics auditor + counterexample |
| [sinr_repair.py](tdma_brain/sinr_repair.py) | SINR-aware schedule repair loop |
| [analyzer.py](tdma_brain/analyzer.py) | δ-robustness + conflict explanations |
| [reporter.py](tdma_brain/reporter.py) | ASCII/HTML/JSON output |
| [adapter.py](emane/adapter.py) | Schedule → EMANE XML bridge |
| [pathloss_bridge.py](emane/pathloss_bridge.py) | Continuous pathloss events |
| [experiment.py](emane/experiment.py) | Automated EMANE experiments |

## Running Tests

```bash
pytest tests/ -v
```
