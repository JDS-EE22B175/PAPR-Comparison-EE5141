"""
Transmitter implementations for all four multicarrier schemes.

Each function takes P QAM symbols and returns the oversampled
time-domain transmit signal.  Block-diagram references point to
the Giridhar GMC++ PDF.

=======================================================================
MATHEMATICAL PROOF: Why F-DOSS ≡ IFDMA (for equi-spaced subcarriers)
=======================================================================

For IFDMA user k=0 with subcarriers {0, K, 2K, …, (P−1)K}:

  Step 1 — P-pt DFT:   D[m] = Σ_{p=0}^{P-1}  d[p] · e^{-j2πmp/P}

  Step 2 — Mapping:     X[mK] = D[m]/√P,   X[other] = 0

  Step 3 — N-pt IFFT:
    x[n] = (1/N) Σ_{l=0}^{N-1} X[l] · e^{j2πln/N}
         = (1/N) Σ_{m=0}^{P-1} D[m]/√P · e^{j2πmKn/N}

    Since K = N/P:
         = 1/(N√P) · Σ_{m=0}^{P-1} D[m] · e^{j2πmn/P}
         = P/(N√P) · IDFT_P{D[m]}  evaluated at  n mod P
         = 1/(K√P) · d[n mod P]

  This is exactly F-DOSS's output (d[n mod P] repeated K times),
  scaled by a constant 1/(K√P).

  PAPR is scale-invariant  →  PAPR_IFDMA ≡ PAPR_FDOSS  ∎

For general user k ≠ 0, the F-DOSS phase ramp e^{j2πkn/N}
corresponds exactly to the frequency shift in IFDMA's subcarrier
indices {k, K+k, …}, so the identity holds for all users.
=======================================================================
"""

import numpy as np
from numpy.fft import fft
from config import N, K, P, L_OS, SCHEME_NAMES, SCHEME_FUNCS
from utils import oversample_freq


# =============================================================================
# OFDMA  (Giridhar PDF page 4)
# =============================================================================
# Block: QAM symbols → S/P → map to P localized subcarriers → N-pt IFFT
#
# Why high PAPR?  Each subcarrier carries an independent QAM symbol.
# The N-pt IFFT sums P independent complex sinusoids → by CLT the
# time-domain envelope is approximately Rayleigh-distributed → tall peaks.
# =============================================================================
def tx_ofdma(symbols_P: np.ndarray) -> np.ndarray:
    """Standard OFDMA: P symbols on P contiguous subcarriers, N-pt IFFT."""
    X = np.zeros(N, dtype=complex)
    X[0:P] = symbols_P
    return oversample_freq(X, N, L_OS)


# =============================================================================
# F-DOSS  (Giridhar PDF page 8; Chang & Chen, IEEE Comm. Lett., Nov 2000)
# =============================================================================
# Block: P QAM symbols → Repeat K times in time → phase ramp → Add CP
#
# NO N-point IFFT!  The transmitter is trivial hardware: a register
# file + counter + CORDIC.  PAPR depends only on the P-symbol sequence
# envelope, not on a superposition of sinusoids.
# =============================================================================
def tx_fdoss(symbols_P: np.ndarray, user_k: int = 0) -> np.ndarray:
    """
    F-DOSS:  s[n] = d[n mod P] · e^{j2πkn/N},  n = 0…N-1.
    For user k=0 the phase = 1 → simple repetition.
    """
    s_time = np.tile(symbols_P, K)
    n = np.arange(N)
    s_time = s_time * np.exp(1j * 2.0 * np.pi * user_k * n / N)
    S_freq = fft(s_time)
    return oversample_freq(S_freq, N, L_OS)


# =============================================================================
# IFDMA  (Giridhar PDF page 11 — "Interleaved OFDMA")
# =============================================================================
# Block: P QAM → P×P DFT ("Mixing Matrix") → distributed subcarriers
#        {k, K+k, 2K+k, …} → N×N IFFT → Add CP
#
# Mathematically identical to F-DOSS for equi-spaced allocation
# (proven in docstring above).  Included separately to demonstrate
# the DFT-precoding pathway and to allow non-equispaced extensions.
# =============================================================================
def tx_ifdma(symbols_P: np.ndarray, user_k: int = 0) -> np.ndarray:
    """
    IFDMA: DFT-precoded symbols on K-spaced (distributed) subcarriers.
    Produces identical PAPR to F-DOSS — see proof in module docstring.
    """
    precoded = fft(symbols_P) / np.sqrt(P)
    X = np.zeros(N, dtype=complex)
    X[user_k + np.arange(P) * K] = precoded
    return oversample_freq(X, N, L_OS)


# =============================================================================
# DFT-spread OFDMA  (SC-FDMA / LFDMA — LTE Uplink)
# =============================================================================
# PDF p.3: "DFT-Precoded OFDMA";  p.7: "3GPP LTE has adopted this for UL"
#
# Same DFT precoding as IFDMA, but with LOCALIZED (contiguous) subcarrier
# mapping.  This breaks the periodic repetition structure → slightly
# higher PAPR than IFDMA/F-DOSS, but gains full scheduling flexibility.
# =============================================================================
def tx_dft_spread_ofdma(symbols_P: np.ndarray) -> np.ndarray:
    """DFT-spread OFDMA (SC-FDMA): DFT-precoded, localized mapping."""
    precoded = fft(symbols_P) / np.sqrt(P)
    X = np.zeros(N, dtype=complex)
    X[0:P] = precoded
    return oversample_freq(X, N, L_OS)


# ─────────────── Register all schemes ───────────────────────────────────────
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
