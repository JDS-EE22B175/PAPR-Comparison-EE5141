"""
Utility functions: QAM constellation generation, PAPR calculation,
oversampling, and CCDF computation.
"""

import numpy as np
from numpy.fft import fft, ifft


# ─────────────────────── QAM Generation ─────────────────────────────────────
def gen_qam_symbols(M: int, num_symbols: int) -> np.ndarray:
    """
    Generate `num_symbols` random M-QAM symbols from a square constellation,
    normalised to unit average power.

    Parameters
    ----------
    M : int
        Constellation size (must be a perfect square: 4, 16, 64, …).
    num_symbols : int
        Number of random symbols to draw.

    Returns
    -------
    np.ndarray of complex128, shape (num_symbols,)
    """
    sqrt_M = int(np.sqrt(M))
    real_vals = np.arange(-(sqrt_M - 1), sqrt_M, 2, dtype=float)
    constellation = (real_vals[:, None] + 1j * real_vals[None, :]).ravel()
    constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))
    indices = np.random.randint(0, M, size=num_symbols)
    return constellation[indices]


# ─────────────────────── Oversampling ───────────────────────────────────────
def oversample_freq(freq_domain_vec: np.ndarray, N_orig: int,
                    L: int) -> np.ndarray:
    """
    Oversample a frequency-domain signal by factor L via zero-padding.

    This is the standard technique (ref: Myung et al., 2006, Sec. III)
    to capture inter-sample peaks.  L ≥ 4 guarantees PAPR accuracy
    within ~0.1 dB of the continuous-time envelope.

    Parameters
    ----------
    freq_domain_vec : array, length N_orig
        Frequency-domain samples (output of FFT).
    N_orig : int
        Original DFT size.
    L : int
        Oversampling factor.

    Returns
    -------
    np.ndarray of complex128, length N_orig * L
        Oversampled time-domain signal (via IFFT of zero-padded spectrum).
    """
    N_os = N_orig * L
    X_os = np.zeros(N_os, dtype=complex)
    half = N_orig // 2
    X_os[:half] = freq_domain_vec[:half]
    X_os[N_os - half:] = freq_domain_vec[half:]
    return ifft(X_os) * np.sqrt(N_os / N_orig)   # energy-normalised


# ─────────────────────── PAPR ───────────────────────────────────────────────
def calc_papr_dB(time_signal: np.ndarray) -> float:
    """
    Compute PAPR in dB.

        PAPR = max|x[n]|² / E[|x[n]|²]
    """
    power = np.abs(time_signal) ** 2
    return 10.0 * np.log10(np.max(power) / np.mean(power))


# ─────────────────────── CCDF ───────────────────────────────────────────────
def compute_ccdf(papr_vals: np.ndarray):
    """
    Return (sorted_papr, ccdf) arrays for plotting.

    CCDF(x) = Pr(PAPR > x) = 1 − CDF(x).
    """
    sorted_papr = np.sort(papr_vals)
    ccdf = 1.0 - np.arange(1, len(sorted_papr) + 1) / len(sorted_papr)
    return sorted_papr, ccdf


def interpolate_papr_at_ccdf(papr_vals: np.ndarray,
                              target_ccdf: float) -> float:
    """Find the PAPR threshold at which CCDF ≈ target_ccdf."""
    sorted_p, ccdf = compute_ccdf(papr_vals)
    idx = np.searchsorted(-ccdf, -target_ccdf)
    return sorted_p[min(idx, len(ccdf) - 1)]
