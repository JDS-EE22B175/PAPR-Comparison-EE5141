"""
Implementation of the four multicarrier transmitters.
Takes P QAM symbols and outputs the oversampled time-domain signal.
References: Giridhar GMC++ presentation slides.
Note: F-DOSS and IFDMA give the exact same PAPR mathematically for
evenly spaced subcarriers. This is expected.
"""

import numpy as np
from numpy.fft import fft
from .config import N, K, P, L_OS, SCHEME_NAMES, SCHEME_FUNCS
from .utils import oversample_freq


# --- OFDMA (Baseline) ---
# Maps P symbols to P contiguous subcarriers, then N-pt IFFT.
# Has high PAPR due to summing many independent sinusoids (Central Limit Theorem).
def tx_ofdma(symbols_P: np.ndarray) -> np.ndarray:
    """Standard OFDMA with localized subcarriers."""
    X = np.zeros(N, dtype=complex)
    X[0:P] = symbols_P
    return oversample_freq(X, N, L_OS)


# --- F-DOSS ---
# Repeats P symbols K times in time domain, adds phase shift.
# Does NOT use an N-point IFFT in the transmitter!
def tx_fdoss(symbols_P: np.ndarray, user_k: int = 0) -> np.ndarray:
    """F-DOSS transmitter. For user k=0, it's just simple repetition."""
    s_time = np.tile(symbols_P, K)
    n = np.arange(N)
    # Apply phase shift for user k
    s_time = s_time * np.exp(1j * 2.0 * np.pi * user_k * n / N)
    S_freq = fft(s_time)
    return oversample_freq(S_freq, N, L_OS)


# --- IFDMA (Interleaved OFDMA) ---
# P-point DFT -> Map to distributed K-spaced subcarriers -> N-point IFFT.
# This gives the same result as F-DOSS algebraically for the time domain.
def tx_ifdma(symbols_P: np.ndarray, user_k: int = 0) -> np.ndarray:
    """IFDMA transmitter with K-spaced distributed subcarriers."""
    precoded = fft(symbols_P) / np.sqrt(P)
    X = np.zeros(N, dtype=complex)
    X[user_k + np.arange(P) * K] = precoded
    return oversample_freq(X, N, L_OS)


# --- DFT-spread OFDMA (SC-FDMA) ---
# Used in LTE uplink. P-point DFT -> contiguous subcarriers -> N-point IFFT.
# Breaks the periodic structure so PAPR is slightly higher than IFDMA,
# but gives more scheduling flexibility.
def tx_dft_spread_ofdma(symbols_P: np.ndarray) -> np.ndarray:
    """DFT-spread OFDMA with localized subcarrier mapping."""
    precoded = fft(symbols_P) / np.sqrt(P)
    X = np.zeros(N, dtype=complex)
    X[0:P] = precoded
    return oversample_freq(X, N, L_OS)


# Register the schemes into the config lists
_REGISTRY = [
    ("OFDMA",         tx_ofdma),
    ("F-DOSS",        tx_fdoss),
    ("IFDMA",         tx_ifdma),
    ("DFT-s-OFDMA",   tx_dft_spread_ofdma),
]

SCHEME_NAMES.clear()
SCHEME_FUNCS.clear()
for name, func in _REGISTRY:
    SCHEME_NAMES.append(name)
    SCHEME_FUNCS.append(func)

