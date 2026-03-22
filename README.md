# Ising Model

This project simulates the 2D Ising model using Monte Carlo methods. Two sampling algorithms are implemented: the Metropolis algorithm and the Heat Bath algorithm. The simulation sweeps across 100 temperature points in the range $k_BT/J \in [0.5, 5]$ on an $L \times L$ square lattice with periodic boundary conditions. At each temperature the system is first equilibrated using a block-averaged convergence criterion, then sampled to compute the energy $E$, magnetization $|M|$, specific heat $C_V$, susceptibility $\chi$, and the fourth-order Binder cumulant $U_4$. Results are compared against the exact Onsager solution for the infinite lattice. The effect of an external magnetic field $B$, antiferromagnetic coupling ($J < 0$), and finite-size scaling across $L = 8, 16, 32$ are also investigated.

## Project Structure

```
.
├── sim.py                          # Metropolis MC simulation (single run)
├── sim_multi.py                    # Metropolis MC simulation (multi-run averaging)
├── sim_with_B.py                   # Metropolis MC with external magnetic field B
├── sim_negative_J.py               # Metropolis MC with antiferromagnetic coupling (J < 0)
├── heat_bath.py                    # Heat Bath MC simulation (single run)
├── heat_bath_multi.py              # Heat Bath MC simulation (multi-run averaging)
├── analytical_sol.py               # Onsager exact solution (E, M, CV)
├── cumulant_test.py                # Binder cumulant crossing plot (Metropolis)
├── cumulant_HB.py                  # Binder cumulant crossing plot (Heat Bath)
├── error_analysis.py               # RMSE residuals: numerical vs analytical
├── benchmark.py                    # Wall-clock timing: Metropolis vs Heat Bath
├── plots_all.py                    # Collective plots for multiple L (Metropolis)
├── plots_B.py                      # Comparison plots for different B values
├── plots_negJ.py                   # Comparison of positive vs negative J
├── plots_heatbath_vs_metropolis.py # Side-by-side HB vs Metropolis comparison
├── demo.py                         # Lattice equilibration snapshots
├── requirements.txt                # Python dependencies
├── data/                           # JSON output files (auto-created)
└── figs/                           # Saved figures (auto-created)
```

## Physical Model

The Hamiltonian is $H = -J \sum_{\langle i,j \rangle} s_i s_j - B \sum_i s_i$ where $s_i \in \{-1, +1\}$, the first sum runs over nearest-neighbour pairs on a square lattice with periodic boundary conditions, $J$ is the exchange coupling, and $B$ is an optional external magnetic field. Temperature enters through the inverse coupling $\beta J = J / (k_BT)$, which is absorbed into $J$ in the code so that $J_\text{code} = 1/T$. Equilibration uses a block-averaged energy convergence check (5 consecutive stable blocks of 100 sweeps each, tolerance $10^{-4}$). All hot-loop kernels (energy difference, sweep, total energy, magnetization) are JIT-compiled with Numba.

## Pipeline

Install the dependencies first:

```bash
pip install -r requirements.txt
```

### Task 1 — Metropolis Algorithm ($B = 0$)

Run the basic Metropolis simulation for a single lattice size:

```bash
python src/sim.py --L 32 --sweeps 10000 --eq 50000 --plot --io
```

For multi-run averaging (reduces statistical noise):

```bash
python src/sim_multi.py --L 32 --runs 3 --seed 0 --plot --io
```

| Flag | Description |
|------|-------------|
| `--L` | Lattice side length (default: 32) |
| `--sweeps` | Number of MC sampling sweeps (default: 10000) |
| `--eq` | Maximum equilibration sweeps (default: 50000) |
| `--runs` | Number of independent runs to average (`sim_multi.py` only, default: 3) |
| `--seed` | Base random seed (`sim_multi.py` only, default: 0) |
| `--plot` | Show the 4-panel plot after the run |
| `--io` | Save results to JSON |

Run for $L = 8, 16, 32$ to enable finite-size comparison and cumulant crossing analysis.

### Task 2 — External Magnetic Field and Negative $J$

Sweep the same temperature range with an external field $B$ applied:

```bash
python src/sim_with_B.py
```

This runs the simulation for $B \in \{0.01, 0.05, 0.1, 0.5, 1.0\}$ and saves each result separately.

For antiferromagnetic coupling ($J < 0$):

```bash
python src/sim_negative_J.py
```

### Task 3 — Heat Bath Algorithm

Run the Heat Bath simulation for a single lattice size:

```bash
python src/heat_bath.py --L 32 --sweeps 10000 --eq 50000 --plot --io
```

For multi-run averaging:

```bash
python src/heat_bath_multi.py --L 32 --runs 3 --seed 0 --plot --io
```

The CLI flags are identical to the Metropolis scripts.

### Analytical Comparison

After running simulations for $L = 8, 16, 32$, compare against the exact Onsager solution:

```bash
python src/analytical_sol.py --energy --io
```

| Flag | Description |
|------|-------------|
| `--energy` | Include the energy panel in the comparison plot |
| `--io` | Save analytical results to JSON |

### Binder Cumulant

Plot the fourth-order Binder cumulant $U_4 = 1 - \langle M^4 \rangle / (3 \langle M^2 \rangle^2)$ for $L = 8, 16, 32$ to locate the critical temperature via the crossing point:

```bash
python src/cumulant_test.py    # Metropolis data
python src/cumulant_HB.py      # Heat Bath data
```

### Error Analysis

Compute RMSE residuals between the numerical (Metropolis and Heat Bath) and analytical solutions:

```bash
python src/error_analysis.py --L 32
```

### Benchmark

Time the Metropolis and Heat Bath algorithms head-to-head:

```bash
python src/benchmark.py --L 32
```

### Plotting

After all simulation data has been generated, produce comparison figures:

```bash
python src/plots_all.py --avg                    # E, M, CV, X for L = 8, 16, 32
python src/plots_B.py --L 32                     # Effect of external field B
python src/plots_negJ.py                         # Positive vs negative J
python src/plots_heatbath_vs_metropolis.py --L 32 # HB vs Metropolis
```

### Lattice Visualization

Generate equilibration snapshots showing the lattice configuration at different stages:

```bash
python src/demo.py --L 32 --T 2.27 --eq 100000
```

| Flag | Description |
|------|-------------|
| `--L` | Lattice side length (default: 32) |
| `--T` | Temperature (default: 2.27, near $T_c$) |
| `--eq` | Number of equilibration sweeps (default: 100000) |
| `--seed` | Random seed (default: 1) |
| `--out` | Output filename (default: auto-generated) |
| `--no-show` | Suppress the figure window |

## Output

All simulation results are saved as JSON files in the `data/` directory. All figures are saved as PNG images in the `figs/` directory. Both directories are created automatically on first run.
