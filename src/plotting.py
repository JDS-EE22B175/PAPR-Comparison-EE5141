"""
Functions to plot the CCDF curves for PAPR comparison.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # For running without a GUI
import matplotlib.pyplot as plt
from .utils import compute_ccdf
from .config import N, P, K, NUM_ITER, SCHEME_NAMES

# basic colors and styles for the 4 schemes
COLORS = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
STYLES = ["-", "-", "--", "-."]

def plot_single_qam(papr_db: np.ndarray, qam_order: int, save_path: str) -> None:
    """Plots CCDF for a single QAM order and saves it to save_path."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for s in range(len(SCHEME_NAMES)):
        sp, ccdf = compute_ccdf(papr_db[s])
        ax.semilogy(sp, ccdf, STYLES[s], color=COLORS[s],
                    linewidth=2.0, label=SCHEME_NAMES[s])

    ax.set_xlabel("PAPR₀ (dB)", fontsize=13)
    ax.set_ylabel("Pr(PAPR > PAPR₀)  [CCDF]", fontsize=13)
    ax.set_title(f"CCDF of PAPR | {qam_order}-QAM, N={N}, P={P}, K={K}, {NUM_ITER} frames",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', alpha=0.4)
    ax.set_ylim([1e-4, 1])
    ax.set_xlim([0, 14])
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

def plot_combined(all_results: dict, qam_orders: list, save_path: str) -> None:
    """Plots side-by-side CCDF subplots for all QAM orders."""
    ncols = len(qam_orders)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols + 1, 7), sharey=True)
    
    # Handle single column case if we only test 1 QAM order
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

    fig.suptitle(f"PAPR CCDF Comparison | N={N}, P={P}, K={K}, {NUM_ITER} Monte Carlo frames",
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")

