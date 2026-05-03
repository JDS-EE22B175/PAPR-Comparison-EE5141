"""
Plotting module — CCDF curves for PAPR comparison.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .utils import compute_ccdf
from .config import N, P, K, NUM_ITER, SCHEME_NAMES

# ──────────── Visual constants ──────────────
COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
STYLES = ["-", "-", "--", "-."]


def plot_single_qam(papr_db: np.ndarray, qam_order: int,
                    save_path: str) -> None:
    """
    Plot CCDF for all schemes at a single QAM order.

    Parameters
    ----------
    papr_db : ndarray, shape (num_schemes, NUM_ITER)
    qam_order : int
    save_path : str — output PNG path
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    for s in range(len(SCHEME_NAMES)):
        sp, ccdf = compute_ccdf(papr_db[s])
        ax.semilogy(sp, ccdf, STYLES[s], color=COLORS[s],
                    linewidth=2.0, label=SCHEME_NAMES[s])

    ax.set_xlabel("PAPR₀ (dB)", fontsize=13)
    ax.set_ylabel("Pr(PAPR > PAPR₀)  [CCDF]", fontsize=13)
    ax.set_title(f"CCDF of PAPR  |  {qam_order}-QAM,  N={N}, P={P},"
                 f" K={K},  {NUM_ITER} frames",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', alpha=0.4)
    ax.set_ylim([1e-4, 1])
    ax.set_xlim([0, 14])
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_combined(all_results: dict, qam_orders: list,
                  save_path: str) -> None:
    """
    Side-by-side CCDF subplots for all QAM orders.

    Parameters
    ----------
    all_results : dict {qam_order: papr_db array}
    qam_orders : list of int
    save_path : str
    """
    ncols = len(qam_orders)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols + 1, 7),
                             sharey=True)
    if ncols == 1:
        axes = [axes]

    for idx, qam in enumerate(qam_orders):
        ax = axes[idx]
        papr_db = all_results[qam]
        for s in range(len(SCHEME_NAMES)):
            sp, ccdf = compute_ccdf(papr_db[s])
            ax.semilogy(sp, ccdf, STYLES[s], color=COLORS[s],
                        linewidth=2.0, label=SCHEME_NAMES[s])
        ax.set_xlabel("PAPR₀ (dB)", fontsize=12)
        ax.set_title(f"{qam}-QAM", fontsize=13, fontweight='bold')
        ax.grid(True, which='both', alpha=0.4)
        ax.set_ylim([1e-4, 1])
        ax.set_xlim([0, 14])
        if idx == 0:
            ax.set_ylabel("Pr(PAPR > PAPR₀)  [CCDF]", fontsize=12)
        ax.legend(fontsize=10, loc='upper right')

    fig.suptitle(f"PAPR CCDF Comparison  |  N={N}, P={P}, K={K},"
                 f"  {NUM_ITER} Monte Carlo frames",
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
