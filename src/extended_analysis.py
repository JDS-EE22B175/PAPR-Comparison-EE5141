"""
Extended analysis beyond the base paper:
  1. PAPR vs FFT size (N sweep) - shows OFDMA PAPR grows with N, GMC stays flat
  2. PAPR vs number of users (K sweep) - shows effect of P=N/K on all schemes
  3. Theoretical OFDMA PAPR bound overlay
  4. Clipping & EVM tradeoff analysis
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

from .utils import gen_qam_symbols, calc_papr_dB, compute_ccdf, oversample_freq
from .config import L_OS, SCHEME_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# 1. PAPR vs N sweep
# ─────────────────────────────────────────────────────────────────────────────

def _build_transmitters(N_val, K_val, P_val, L_val):
    """
    Builds local transmitter functions for arbitrary N, K, P, L values.
    This avoids depending on the global config for sweep analysis.
    """
    def _oversample(freq_vec, N_orig, L):
        N_os = N_orig * L
        X_os = np.zeros(N_os, dtype=complex)
        half = N_orig // 2
        X_os[:half] = freq_vec[:half]
        X_os[N_os - half:] = freq_vec[half:]
        return ifft(X_os) * np.sqrt(N_os / N_orig)

    def tx_ofdma(syms):
        X = np.zeros(N_val, dtype=complex)
        X[0:P_val] = syms
        return _oversample(X, N_val, L_val)

    def tx_fdoss(syms):
        s_time = np.tile(syms, K_val)
        S_freq = fft(s_time)
        return _oversample(S_freq, N_val, L_val)

    def tx_ifdma(syms):
        precoded = fft(syms) / np.sqrt(P_val)
        X = np.zeros(N_val, dtype=complex)
        X[np.arange(P_val) * K_val] = precoded
        return _oversample(X, N_val, L_val)

    def tx_dft_s_ofdma(syms):
        precoded = fft(syms) / np.sqrt(P_val)
        X = np.zeros(N_val, dtype=complex)
        X[0:P_val] = precoded
        return _oversample(X, N_val, L_val)

    return [tx_ofdma, tx_fdoss, tx_ifdma, tx_dft_s_ofdma]


def papr_vs_N_sweep(
    N_values: list = None,
    K_fixed: int = 4,
    qam_order: int = 16,
    num_iter: int = 5000,
) -> dict:
    """
    Sweeps over different FFT sizes N, keeping K fixed.
    P = N/K varies with N.
    Returns dict: {N: array of shape (4, num_iter)}.
    """
    if N_values is None:
        N_values = [32, 64, 128, 256, 512, 1024]

    results = {}
    for N_val in N_values:
        P_val = N_val // K_fixed
        print(f"  N={N_val}, P={P_val} ...", end=" ", flush=True)
        
        txs = _build_transmitters(N_val, K_fixed, P_val, L_OS)
        papr_db = np.zeros((4, num_iter))
        
        for i in range(num_iter):
            syms = gen_qam_symbols(qam_order, P_val)
            for s, tx_fn in enumerate(txs):
                sig = tx_fn(syms)
                papr_db[s, i] = calc_papr_dB(sig)
        
        results[N_val] = papr_db
        means = [np.mean(papr_db[s]) for s in range(4)]
        print(f"means: {[f'{m:.2f}' for m in means]}")
    
    return results


def plot_papr_vs_N(results: dict, save_path: str) -> None:
    """
    Plots mean PAPR and 99th percentile PAPR vs N for all four schemes.
    Two subplots: (a) Mean PAPR  (b) 99th percentile.
    """
    N_vals = sorted(results.keys())
    
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    markers = ["o", "s", "D", "^"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for s, name in enumerate(SCHEME_NAMES):
        means = [np.mean(results[N][s]) for N in N_vals]
        p99s = [np.percentile(results[N][s], 99) for N in N_vals]
        
        axes[0].plot(N_vals, means, marker=markers[s], color=colors[s],
                     linewidth=2, markersize=8, label=name)
        axes[1].plot(N_vals, p99s, marker=markers[s], color=colors[s],
                     linewidth=2, markersize=8, label=name)
    
    axes[0].set_xlabel("FFT Size N", fontsize=13)
    axes[0].set_ylabel("Mean PAPR (dB)", fontsize=13)
    axes[0].set_title("Mean PAPR vs FFT Size", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.4)
    axes[0].set_xscale('log', base=2)
    axes[0].set_xticks(N_vals)
    axes[0].set_xticklabels([str(n) for n in N_vals])
    
    axes[1].set_xlabel("FFT Size N", fontsize=13)
    axes[1].set_ylabel("99th Percentile PAPR (dB)", fontsize=13)
    axes[1].set_title("PAPR @ CCDF=1% vs FFT Size", fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.4)
    axes[1].set_xscale('log', base=2)
    axes[1].set_xticks(N_vals)
    axes[1].set_xticklabels([str(n) for n in N_vals])
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. PAPR vs K sweep (number of users)
# ─────────────────────────────────────────────────────────────────────────────

def papr_vs_K_sweep(
    K_values: list = None,
    N_fixed: int = 256,
    qam_order: int = 16,
    num_iter: int = 5000,
) -> dict:
    """
    Sweeps over K (number of users), keeping N fixed.
    P = N/K shrinks as K grows → fewer tones per user → different PAPR behavior.
    """
    if K_values is None:
        # K must divide N evenly
        K_values = [k for k in [1, 2, 4, 8, 16, 32, 64] if N_fixed % k == 0]
    
    results = {}
    for K_val in K_values:
        P_val = N_fixed // K_val
        if P_val < 2:
            continue
        print(f"  K={K_val}, P={P_val} ...", end=" ", flush=True)
        
        txs = _build_transmitters(N_fixed, K_val, P_val, L_OS)
        papr_db = np.zeros((4, num_iter))
        
        for i in range(num_iter):
            syms = gen_qam_symbols(qam_order, P_val)
            for s, tx_fn in enumerate(txs):
                sig = tx_fn(syms)
                papr_db[s, i] = calc_papr_dB(sig)
        
        results[K_val] = papr_db
        means = [np.mean(papr_db[s]) for s in range(4)]
        print(f"means: {[f'{m:.2f}' for m in means]}")
    
    return results


def plot_papr_vs_K(results: dict, N_fixed: int, save_path: str) -> None:
    """
    Plots mean PAPR and 99th percentile vs K.
    Also shows P=N/K on a secondary x-axis for clarity.
    """
    K_vals = sorted(results.keys())
    P_vals = [N_fixed // k for k in K_vals]
    
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    markers = ["o", "s", "D", "^"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for s, name in enumerate(SCHEME_NAMES):
        means = [np.mean(results[K][s]) for K in K_vals]
        p99s = [np.percentile(results[K][s], 99) for K in K_vals]
        
        axes[0].plot(K_vals, means, marker=markers[s], color=colors[s],
                     linewidth=2, markersize=8, label=name)
        axes[1].plot(K_vals, p99s, marker=markers[s], color=colors[s],
                     linewidth=2, markersize=8, label=name)
    
    for ax_idx, (ax, ylabel, title) in enumerate(zip(
        axes,
        ["Mean PAPR (dB)", "99th Percentile PAPR (dB)"],
        ["Mean PAPR vs Users (K)", "PAPR @ CCDF=1% vs Users (K)"]
    )):
        ax.set_xlabel("Number of Users K", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.4)
        ax.set_xscale('log', base=2)
        ax.set_xticks(K_vals)
        ax.set_xticklabels([str(k) for k in K_vals])
        
        # Secondary axis showing P = N/K
        ax2 = ax.twiny()
        ax2.set_xscale('log', base=2)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(K_vals)
        ax2.set_xticklabels([f"P={p}" for p in P_vals], fontsize=9, color='gray')
        ax2.set_xlabel("Subcarriers per User (P = N/K)", fontsize=10, color='gray')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Theoretical OFDMA PAPR bound overlay
# ─────────────────────────────────────────────────────────────────────────────

def theoretical_ofdma_ccdf(papr0_dB: np.ndarray, N_sub: int) -> np.ndarray:
    """
    Analytical approximation for OFDMA CCDF:
      Pr(PAPR > γ) ≈ 1 - (1 - exp(-γ))^N
    
    This assumes N independent Rayleigh-distributed samples (CLT approximation).
    Reference: Ochiai & Imai, IEEE Trans. Comm., 2001.
    
    Args:
        papr0_dB: PAPR thresholds in dB
        N_sub: number of subcarriers (controls how tight the bound is)
    
    Returns:
        ccdf: Pr(PAPR > papr0) for each threshold
    """
    gamma = 10.0 ** (papr0_dB / 10.0)  # Convert dB to linear
    ccdf = 1.0 - (1.0 - np.exp(-gamma)) ** N_sub
    return np.clip(ccdf, 1e-10, 1.0)


def plot_ccdf_with_theory(
    papr_db: np.ndarray,
    qam_order: int,
    N_sub: int,
    save_path: str,
) -> None:
    """
    Plots the empirical CCDF for all 4 schemes with the theoretical
    OFDMA bound overlaid as a dashed black curve.
    """
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    styles = ["-", "-", "--", "-."]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot empirical curves
    for s, name in enumerate(SCHEME_NAMES):
        sp, ccdf = compute_ccdf(papr_db[s])
        ax.semilogy(sp, ccdf, styles[s], color=colors[s],
                    linewidth=2.0, label=name)
    
    # Overlay theoretical bound
    papr0_range = np.linspace(0, 14, 500)
    theory_ccdf = theoretical_ofdma_ccdf(papr0_range, N_sub)
    ax.semilogy(papr0_range, theory_ccdf, 'k--', linewidth=2.0, alpha=0.7,
                label=f"Theory (N={N_sub})")
    
    ax.set_xlabel("PAPR₀ (dB)", fontsize=13)
    ax.set_ylabel("Pr(PAPR > PAPR₀)  [CCDF]", fontsize=13)
    ax.set_title(
        f"CCDF with Theoretical Bound | {qam_order}-QAM, N={N_sub}",
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which='both', alpha=0.4)
    ax.set_ylim([1e-4, 1])
    ax.set_xlim([0, 14])
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Clipping & EVM Tradeoff
# ─────────────────────────────────────────────────────────────────────────────

def clip_signal(time_signal: np.ndarray, clip_ratio: float) -> np.ndarray:
    """
    Clips a time-domain signal at clip_ratio * rms_amplitude.
    clip_ratio is the Clipping Ratio (CR) in linear scale.
    
    CR = A_clip / sigma, where sigma = RMS amplitude.
    Lower CR = more aggressive clipping = lower PAPR but higher distortion.
    """
    rms = np.sqrt(np.mean(np.abs(time_signal) ** 2))
    threshold = clip_ratio * rms
    
    magnitude = np.abs(time_signal)
    phase = np.angle(time_signal)
    
    clipped_mag = np.minimum(magnitude, threshold)
    return clipped_mag * np.exp(1j * phase)


def compute_evm(original: np.ndarray, distorted: np.ndarray) -> float:
    """
    Computes Error Vector Magnitude (EVM) as a percentage.
    EVM = sqrt(mean(|x - x_hat|^2) / mean(|x|^2)) * 100
    """
    error_power = np.mean(np.abs(original - distorted) ** 2)
    signal_power = np.mean(np.abs(original) ** 2)
    return np.sqrt(error_power / signal_power) * 100.0


def clipping_analysis(
    qam_order: int = 16,
    N_val: int = 256,
    K_val: int = 4,
    num_iter: int = 3000,
    cr_values: list = None,
) -> dict:
    """
    Analyzes the PAPR vs EVM tradeoff for OFDMA with different clipping ratios.
    
    Returns dict with keys:
      'cr_values': list of clipping ratios
      'mean_papr': mean PAPR for each CR
      'p99_papr': 99th percentile PAPR for each CR
      'mean_evm': mean EVM (%) for each CR
    """
    if cr_values is None:
        cr_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 100.0]
    
    P_val = N_val // K_val
    txs = _build_transmitters(N_val, K_val, P_val, L_OS)
    tx_ofdma = txs[0]  # OFDMA only
    
    mean_papr = []
    p99_papr = []
    mean_evm = []
    
    for cr in cr_values:
        paprs = np.zeros(num_iter)
        evms = np.zeros(num_iter)
        
        for i in range(num_iter):
            syms = gen_qam_symbols(qam_order, P_val)
            sig = tx_ofdma(syms)
            
            clipped = clip_signal(sig, cr)
            paprs[i] = calc_papr_dB(clipped)
            evms[i] = compute_evm(sig, clipped)
        
        mean_papr.append(np.mean(paprs))
        p99_papr.append(np.percentile(paprs, 99))
        mean_evm.append(np.mean(evms))
        
        cr_label = f"CR={cr:.1f}" if cr < 50 else "No clip"
        print(f"  {cr_label}: PAPR_mean={np.mean(paprs):.2f}dB, "
              f"PAPR_99%={np.percentile(paprs, 99):.2f}dB, "
              f"EVM={np.mean(evms):.2f}%")
    
    return {
        'cr_values': cr_values,
        'mean_papr': mean_papr,
        'p99_papr': p99_papr,
        'mean_evm': mean_evm,
    }


def plot_clipping_tradeoff(clip_results: dict, save_path: str) -> None:
    """
    Plots the PAPR reduction vs EVM degradation tradeoff.
    Two subplots: (a) PAPR vs CR  (b) EVM vs CR  with a Pareto-style annotation.
    """
    cr = clip_results['cr_values']
    # Filter out the "no clip" point for cleaner plots
    mask = [c < 50 for c in cr]
    cr_plot = [c for c, m in zip(cr, mask) if m]
    p99_plot = [p for p, m in zip(clip_results['p99_papr'], mask) if m]
    mean_plot = [p for p, m in zip(clip_results['mean_papr'], mask) if m]
    evm_plot = [e for e, m in zip(clip_results['mean_evm'], mask) if m]
    
    # Also get the "no clip" baseline
    no_clip_papr = clip_results['p99_papr'][-1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # (a) PAPR vs Clipping Ratio
    axes[0].plot(cr_plot, p99_plot, 'o-', color="#e74c3c", linewidth=2,
                 markersize=8, label="99th percentile")
    axes[0].plot(cr_plot, mean_plot, 's--', color="#3498db", linewidth=2,
                 markersize=7, label="Mean")
    axes[0].axhline(y=no_clip_papr, color='gray', linestyle=':', alpha=0.6,
                    label=f"Unclipped baseline ({no_clip_papr:.1f} dB)")
    axes[0].set_xlabel("Clipping Ratio (CR = A_clip / σ)", fontsize=13)
    axes[0].set_ylabel("PAPR (dB)", fontsize=13)
    axes[0].set_title("PAPR Reduction via Clipping", fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.4)
    
    # (b) EVM vs Clipping Ratio
    axes[1].plot(cr_plot, evm_plot, 'D-', color="#9b59b6", linewidth=2,
                 markersize=8)
    axes[1].set_xlabel("Clipping Ratio (CR = A_clip / σ)", fontsize=13)
    axes[1].set_ylabel("EVM (%)", fontsize=13)
    axes[1].set_title("Signal Distortion (EVM) from Clipping", fontsize=13,
                      fontweight='bold')
    axes[1].grid(True, alpha=0.4)
    
    # Annotate the "sweet spot" region (CR ≈ 2-3)
    axes[1].axvspan(1.8, 3.0, alpha=0.1, color='green')
    axes[1].text(2.4, max(evm_plot) * 0.5, "Sweet spot\n(CR ≈ 2–3)",
                 ha='center', fontsize=11, color='green', fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_clipping_ccdf(
    qam_order: int = 16,
    N_val: int = 256,
    K_val: int = 4,
    num_iter: int = 5000,
    cr_values: list = None,
    save_path: str = "outputs/clipping_ccdf.png",
) -> None:
    """
    Overlays CCDF curves for OFDMA at different clipping ratios
    to visually show how the tail shrinks with clipping.
    """
    if cr_values is None:
        cr_values = [1.4, 2.0, 3.0, 100.0]  # Selected CRs + no-clip
    
    P_val = N_val // K_val
    txs = _build_transmitters(N_val, K_val, P_val, L_OS)
    tx_ofdma = txs[0]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    cr_colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
    
    for idx, cr in enumerate(cr_values):
        paprs = np.zeros(num_iter)
        for i in range(num_iter):
            syms = gen_qam_symbols(qam_order, P_val)
            sig = tx_ofdma(syms)
            clipped = clip_signal(sig, cr)
            paprs[i] = calc_papr_dB(clipped)
        
        sp, ccdf = compute_ccdf(paprs)
        label = f"CR={cr:.1f}" if cr < 50 else "No clipping"
        ax.semilogy(sp, ccdf, linewidth=2.0, color=cr_colors[idx], label=label)
    
    ax.set_xlabel("PAPR₀ (dB)", fontsize=13)
    ax.set_ylabel("Pr(PAPR > PAPR₀)  [CCDF]", fontsize=13)
    ax.set_title(f"OFDMA CCDF at Various Clipping Ratios | {qam_order}-QAM, N={N_val}",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, which='both', alpha=0.4)
    ax.set_ylim([1e-4, 1])
    ax.set_xlim([0, 14])
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
