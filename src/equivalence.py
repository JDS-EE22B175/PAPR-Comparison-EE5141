"""
Verification that F-DOSS and IFDMA produce equivalent time-domain signals.

Mathematical proof sketch:
  F-DOSS: takes P symbols d[0..P-1], tiles them K times in time domain.
    x_fdoss[n] = d[n mod P] · exp(j2πkn/N)   (for user k)
    
  IFDMA:  P-pt DFT of symbols → map to K-spaced subcarriers → N-pt IFFT.
    D[m] = DFT_P{d[n]},  then place D[m] at subcarriers m*K+k.
    
  Key identity: The N-pt FFT of a K-fold tiled sequence d[n mod P] produces
  non-zero coefficients ONLY at indices that are multiples of K:
    FFT_N{tile(d, K)}[m*K] = K · DFT_P{d}[m],   for m = 0, ..., P-1
  
  So F-DOSS's frequency-domain representation IS the K-spaced subcarrier
  mapping — just with a scale factor of K*sqrt(P) instead of 1/sqrt(P).
  
  The PAPR is identical because PAPR = max/mean is scale-invariant.
  The actual waveforms differ only by a constant scale factor.
  
  Result: ||x_fdoss / ||x_fdoss|| - x_ifdma / ||x_ifdma||| ≈ 0 (machine precision).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import N, K, P
from .transmitters import tx_fdoss, tx_ifdma
from .utils import gen_qam_symbols, calc_papr_dB


def verify_equivalence(qam_order: int = 16, num_trials: int = 100) -> dict:
    """
    Runs multiple trials comparing F-DOSS and IFDMA output signals.
    
    Since the two transmitters use different internal scaling conventions
    (F-DOSS: tile then FFT; IFDMA: DFT/sqrt(P) then place), we compare:
      1. Normalized waveforms (should match to machine precision)
      2. PAPR values (should be identical since PAPR is scale-invariant)
    """
    shape_errors = np.zeros(num_trials)
    papr_diffs = np.zeros(num_trials)
    
    for i in range(num_trials):
        syms = gen_qam_symbols(qam_order, P)
        sig_fdoss = tx_fdoss(syms, user_k=0)
        sig_ifdma = tx_ifdma(syms, user_k=0)
        
        # Normalize both to unit energy for shape comparison
        norm_fdoss = sig_fdoss / np.sqrt(np.mean(np.abs(sig_fdoss)**2))
        norm_ifdma = sig_ifdma / np.sqrt(np.mean(np.abs(sig_ifdma)**2))
        
        shape_errors[i] = np.max(np.abs(norm_fdoss - norm_ifdma))
        
        # PAPR comparison (scale-invariant, so no normalization needed)
        papr_diffs[i] = abs(calc_papr_dB(sig_fdoss) - calc_papr_dB(sig_ifdma))
    
    return {
        "shape_errors": shape_errors,
        "papr_diffs": papr_diffs,
        "mean_shape_error": np.mean(shape_errors),
        "worst_shape_error": np.max(shape_errors),
        "mean_papr_diff": np.mean(papr_diffs),
        "worst_papr_diff": np.max(papr_diffs),
        "shapes_match": bool(np.all(shape_errors < 1e-10)),
        "papr_match": bool(np.all(papr_diffs < 1e-10)),
    }


def plot_equivalence(qam_order: int, save_path: str) -> None:
    """
    Generates a publication-quality figure showing:
      (a) F-DOSS time-domain signal magnitude (normalized)
      (b) IFDMA time-domain signal magnitude (normalized)
      (c) Absolute difference of normalized signals (~1e-15)
      (d) Inset annotation of the K-fold periodic structure
    """
    np.random.seed(123)
    syms = gen_qam_symbols(qam_order, P)
    
    sig_fdoss = tx_fdoss(syms, user_k=0)
    sig_ifdma = tx_ifdma(syms, user_k=0)
    
    # Normalize for shape comparison
    norm_fdoss = sig_fdoss / np.sqrt(np.mean(np.abs(sig_fdoss)**2))
    norm_ifdma = sig_ifdma / np.sqrt(np.mean(np.abs(sig_ifdma)**2))
    
    N_os = len(sig_fdoss)
    t = np.arange(N_os)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # (a) F-DOSS (normalized)
    axes[0].plot(t, np.abs(norm_fdoss), color="#2ecc71", linewidth=0.8, alpha=0.9)
    axes[0].set_ylabel("|x(n)| (normalized)", fontsize=12)
    axes[0].set_title(f"F-DOSS Time-Domain Signal  ({qam_order}-QAM, N={N}, P={P})",
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Annotate the K-fold periodicity
    period = N_os // K
    for rep in range(K):
        axes[0].axvline(x=rep * period, color='gray', linestyle=':', alpha=0.5)
    axes[0].text(period / 2, axes[0].get_ylim()[1] * 0.85, "← P samples (repeated K times) →",
                 ha='center', fontsize=9, color='gray')
    
    # (b) IFDMA (normalized)
    axes[1].plot(t, np.abs(norm_ifdma), color="#3498db", linewidth=0.8, alpha=0.9)
    axes[1].set_ylabel("|x(n)| (normalized)", fontsize=12)
    axes[1].set_title("IFDMA Time-Domain Signal", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    for rep in range(K):
        axes[1].axvline(x=rep * period, color='gray', linestyle=':', alpha=0.5)
    
    # (c) Difference
    diff = np.abs(norm_fdoss - norm_ifdma)
    axes[2].plot(t, diff, color="#e74c3c", linewidth=0.8)
    axes[2].set_ylabel("|Δx(n)|", fontsize=12)
    axes[2].set_xlabel("Sample index n", fontsize=12)
    axes[2].set_title(
        f"Absolute Difference (max = {np.max(diff):.2e}  — machine precision)",
        fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    
    # PAPR comparison annotation
    papr_fdoss = calc_papr_dB(sig_fdoss)
    papr_ifdma = calc_papr_dB(sig_ifdma)
    fig.text(0.5, -0.02,
             f"PAPR: F-DOSS = {papr_fdoss:.6f} dB,  IFDMA = {papr_ifdma:.6f} dB  "
             f"(Δ = {abs(papr_fdoss - papr_ifdma):.2e} dB)",
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")
