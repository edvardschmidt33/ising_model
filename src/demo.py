import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

# Import from your sim.py
from sim import init_lattice, MonterCarlo_move, Energy, Mag



def make_snapshots(
    L: int = 32,
    T: float = 2.27,
    eq_sweeps: int = 50000,
    snaps: int = 6,
    sweeps_per_step: int = 1,
    seed: int | None = 0,
):
    """
    Runs Metropolis equilibration and captures `snaps` lattice snapshots from
    sweep 0 up to eq_sweeps.

    sweeps_per_step:
        how many Monte Carlo sweeps to apply between sweep counter increments
        (keep at 1 if MonterCarlo_move already equals 1 sweep).
    """
    if seed is not None:
        np.random.seed(seed)

    J = 1.0 / T  # your convention (beta baked into J)
    N = L * L

    s = init_lattice(L)

    # Choose 6 stages: 0%, 1%, 5%, 10%, 25%, 100% of equilibration
    frac = np.array([0.0, 0.01, 0.05, 0.10, 0.25, 1.0])
    sweep_targets = (frac * eq_sweeps).astype(int)

    # Ensure strictly increasing and within bounds
    sweep_targets = np.unique(np.clip(sweep_targets, 0, eq_sweeps))

    # If unique reduced count (small eq_sweeps), pad with linear targets
    if len(sweep_targets) < snaps:
        sweep_targets = np.linspace(0, eq_sweeps, snaps).astype(int)
        sweep_targets = np.unique(sweep_targets)

    # If too many, keep first `snaps` (shouldn’t happen with default frac)
    sweep_targets = sweep_targets[:snaps]

    snapshots = []
    current = 0

    # Record sweep 0
    snapshots.append((0, s.copy(), Energy(s, L, J) / N, abs(Mag(s)) / N))

    for target in sweep_targets[1:]:
        while current < target:
            # one "sweep" of Metropolis according to your sim.py function
            for _ in range(sweeps_per_step):
                MonterCarlo_move(s, L, J)
            current += 1

        e = Energy(s, L, J) / N
        m = abs(Mag(s)) / N
        snapshots.append((current, s.copy(), e, m))

    return snapshots

def plot_snapshots(snapshots, L: int, T: float, outpath: str | None = None, show: bool = True):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.ravel()

    for ax, (sw, conf, e, m) in zip(axes, snapshots):
        ax.imshow(conf, cmap="Blues", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(f"{sw}")
        ax.set_xticks([])
        ax.set_yticks([])

    # If fewer than 6 snapshots, hide remaining axes
    for k in range(len(snapshots), 6):
        axes[k].axis("off")

    fig.suptitle(f"Metropolis equilibration snapshots (L={L}, T={T})", fontsize=14)
    fig.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=300)

    if show:
        plt.show()

    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapshot demo: Metropolis equilibration (2D Ising)")
    parser.add_argument("--L", type=int, default=32, help="Lattice size")
    parser.add_argument("--T", type=float, default=2.27, help="Temperature")
    parser.add_argument("--eq", type=int, default=100000, help="Equilibration sweeps")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--out", type=str, default="", help="Output filename (png). If empty, auto-name.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the figure window")

    args = parser.parse_args()

    snaps = make_snapshots(L=args.L, T=args.T, eq_sweeps=args.eq, seed=args.seed)

    out = args.out.strip()
    if not out:
        out = f"./figs/equilibration_snapshots_L{args.L}_T{args.T:.3f}.png"

    plot_snapshots(snaps, L=args.L, T=args.T, outpath=out, show=(not args.no_show))
    print(f"Saved snapshots figure to: {out}")