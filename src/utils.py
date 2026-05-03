"""
Utility functions for QAM generation, PAPR, and CCDF calculation.
"""

import numpy as np
from numpy.fft import fft, ifft

# --- QAM Generation ---
def gen_qam_symbols(M: int, num_symbols: int) -> np.ndarray:
    """
    Generates random M-QAM symbols with average power of 1.
    M must be a perfect square like 4, 16, 64.
    """
    sqrt_M = int(np.sqrt(M))
    real_vals = np.arange(-(sqrt_M - 1), sqrt_M, 2, dtype=float)
    constellation = (real_vals[:, None] + 1j * real_vals[None, :]).ravel()
    
    # normalize power
    constellation /= np.sqrt(np.mean(np.abs(constellation) ** 2))
    indices = np.random.randint(0, M, size=num_symbols)
    return constellation[indices]

# --- Oversampling ---
def oversample_freq(freq_domain_vec: np.ndarray, N_orig: int, L: int) -> np.ndarray:
    """
    Oversamples a frequency-domain signal by factor L by zero padding.
    L=4 is usually enough for accurate PAPR.
    """
    N_os = N_orig * L
    X_os = np.zeros(N_os, dtype=complex)
    half = N_orig // 2
    
    X_os[:half] = freq_domain_vec[:half]
    X_os[N_os - half:] = freq_domain_vec[half:]
    
    return ifft(X_os) * np.sqrt(N_os / N_orig)   # scaling to maintain energy

# --- PAPR ---
def calc_papr_dB(time_signal: np.ndarray) -> float:
    """Calculates PAPR in dB: max(|x|^2) / mean(|x|^2)"""
    power = np.abs(time_signal) ** 2
    return 10.0 * np.log10(np.max(power) / np.mean(power))

# --- CCDF ---
def compute_ccdf(papr_vals: np.ndarray):
    """Returns sorted PAPR values and their CCDF probabilities."""
    sorted_papr = np.sort(papr_vals)
    ccdf = 1.0 - np.arange(1, len(sorted_papr) + 1) / len(sorted_papr)
    return sorted_papr, ccdf

def interpolate_papr_at_ccdf(papr_vals: np.ndarray, target_ccdf: float) -> float:
    """Finds the PAPR value at a specific CCDF threshold (like 1% or 0.1%)."""
    sorted_p, ccdf = compute_ccdf(papr_vals)
    idx = np.searchsorted(-ccdf, -target_ccdf)
    return sorted_p[min(idx, len(ccdf) - 1)]

