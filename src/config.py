"""
Global simulation parameters.
Reference: Giridhar PDF pages 7-11.

  N = K × P   where:
    N = total system subcarriers (FFT size)
    K = number of uplink users
    P = subcarriers allocated per user
"""

# ──────────── System Parameters ──────────────
N = 256          # Total subcarriers / FFT size
K = 4            # Number of uplink users
P = N // K       # Subcarriers per user  (= 64)

assert K * P == N, f"K*P = {K*P} ≠ N = {N}"

# ──────────── Simulation Parameters ──────────
NUM_ITER   = 10_000        # Monte Carlo iterations per QAM order
L_OS       = 4             # Oversampling factor (L≥4 → <0.1 dB PAPR error)
QAM_ORDERS = [4, 16, 64]   # QPSK, 16-QAM, 64-QAM
RANDOM_SEED = 42            # For reproducibility

# ──────────── Scheme Registry ────────────────
# Filled by transmitters.py at import time
SCHEME_NAMES = []
SCHEME_FUNCS = []
