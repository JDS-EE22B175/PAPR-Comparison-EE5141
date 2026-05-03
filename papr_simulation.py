"""
=============================================================================
PAPR Comparison: OFDMA vs F-DOSS vs IFDMA vs DFT-spread OFDMA
=============================================================================
Reference : "Generalised Multi-Carrier (GMC) ++" – K. Giridhar, IIT Madras
Course    : EE5141 – Wireless & Cellular Communications
Metric    : CCDF of Peak-to-Average Power Ratio (PAPR)

Notation (from the PDF, pages 7-11):
  N = total FFT size  (system bandwidth)
  K = number of uplink users
  P = subcarriers per user   =>  K * P = N

Block-diagram mapping (PDF page numbers noted inline).
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# ─────────────────────────── Global Parameters ──────────────────────────────
N = 256          # Total subcarriers (FFT size)
K = 4            # Number of uplink users
P = N // K       # Subcarriers per user  (P = 64)
NUM_ITER = 10000 # Monte-Carlo iterations
QAM_ORDER = 16   # 16-QAM constellation
L_OS = 4         # Oversampling factor for accurate PAPR measurement

assert K * P == N, "K*P must equal N"

# ─────────────────────────── QAM Constellation ──────────────────────────────
def gen_qam_symbols(M, num_symbols):
    """Generate random M-QAM symbols (Gray-coded constellation points)."""
    sqrt_M = int(np.sqrt(M))
    real_vals = np.arange(-(sqrt_M - 1), sqrt_M, 2)  # e.g. [-3,-1,1,3]
    constellation = (real_vals[:, None] + 1j * real_vals[None, :]).ravel()
    constellation /= np.sqrt(np.mean(np.abs(constellation)**2))  # unit avg power
    indices = np.random.randint(0, M, size=num_symbols)
    return constellation[indices]

# ─────────────────────────── PAPR Calculation ───────────────────────────────
def calc_papr_dB(time_signal):
    """
    PAPR = max|x(n)|^2 / E[|x(n)|^2]   (in dB)
    The signal is already oversampled for accurate peak capture.
    """
    power = np.abs(time_signal)**2
    papr_linear = np.max(power) / np.mean(power)
    return 10 * np.log10(papr_linear)

def oversample_freq(freq_domain_vec, N_orig, L):
    """
    Zero-pad in frequency domain to achieve L× oversampling.
    Input : freq_domain_vec of length N_orig
    Output: time-domain signal of length N_orig * L  (via IFFT)
    """
    N_os = N_orig * L
    X_os = np.zeros(N_os, dtype=complex)
    # Place original spectrum: DC … N/2-1 at the start, N/2 … N-1 at the end
    half = N_orig // 2
    X_os[:half] = freq_domain_vec[:half]
    X_os[N_os - half:] = freq_domain_vec[half:]
    return ifft(X_os) * np.sqrt(N_os / N_orig)  # energy-normalised

# =============================================================================
# TRANSMITTER 1 : Baseline OFDMA  (PDF page 4)
# =============================================================================
# Block diagram: QAM symbols → S/P → map to P assigned subcarriers → N-pt IFFT
# This is standard OFDMA where one user occupies P contiguous subcarriers.
# =============================================================================
def tx_ofdma(symbols_P):
    """
    Standard OFDMA: map P symbols to P contiguous subcarriers out of N,
    then N-point IFFT.  Uses oversampling for accurate PAPR.
    """
    X = np.zeros(N, dtype=complex)
    X[0:P] = symbols_P                # Localized allocation (first P carriers)
    return oversample_freq(X, N, L_OS) # → time-domain via L×-oversampled IFFT

# =============================================================================
# TRANSMITTER 2 : F-DOSS  (PDF page 8 – "Freq-Domain Orthogonal Spread Spectrum")
# =============================================================================
# Block diagram: Take P QAM symbols → Repeat K times in time
#                → multiply by user-specific phase ramp  e^{j·2π·k·n/N}
#                → Add CP
#
# KEY INSIGHT (from PDF page 10):
#   "Low PAPR" because the time-domain signal for user k is just K copies
#   of the P QAM symbols with a phase rotation — NO N-point IFFT involved.
#   The signal is essentially single-carrier-like within each repetition.
#
# For PAPR measurement we model user #0 (k=0), so phase = e^{j0} = 1.
# The time-domain signal is simply the P symbols repeated K times.
# =============================================================================
def tx_fdoss(symbols_P, user_k=0):
    """
    F-DOSS transmitter (PDF page 8):
      s(n) = symbols_P[n mod P] * exp(j*2*pi*k*n/N),  n = 0..N-1

    For user k=0 the phase term is 1, so the output is P symbols
    tiled K times — a truly single-carrier-like waveform.
    """
    # Step 1: Repeat P symbols K times in time  (PDF: "Repeat K times")
    s_time = np.tile(symbols_P, K)   # length N = K*P

    # Step 2: User-specific phase ramp  (PDF: e^{j 2π k n / N})
    n = np.arange(N)
    phase = np.exp(1j * 2 * np.pi * user_k * n / N)
    s_time = s_time * phase

    # Oversample in freq domain for accurate PAPR measurement
    S_freq = fft(s_time)
    return oversample_freq(S_freq, N, L_OS)

# =============================================================================
# TRANSMITTER 3 : Interleaved OFDMA / IFDMA  (PDF page 11)
# =============================================================================
# Block diagram: Take P QAM symbols
#       → P×P Mixing Matrix  (= P-point DFT, see PDF: "Mixing Matrix can
#                              simply be a K-point DFT matrix!")
#       → Place on K-spaced (distributed/interleaved) subcarriers:
#                sub-carriers  {k, K+k, 2K+k, ..., (P-1)K+k}  for user k
#       → N×N IFFT → Add CP
#
# PDF page 10: "Some PAPR increase + increase in computational complexity,
#               but ensures more flexibility"
# =============================================================================
def tx_ifdma(symbols_P, user_k=0):
    """
    IFDMA transmitter (PDF page 11):
      1) P-pt DFT precoding  (the "Mixing Matrix")
      2) Map to distributed subcarriers spaced K apart
      3) N-pt IFFT
    """
    # Step 1: P×P DFT mixing matrix  (PDF: "Mixing Matrix = K-pt DFT")
    #   NOTE: PDF says "K-point DFT" but the matrix operates on P symbols,
    #   and when K==P the dimensions match. We use P-pt DFT on the P symbols.
    precoded = fft(symbols_P) / np.sqrt(P)   # P-point DFT (normalised)

    # Step 2: Map to distributed (interleaved) subcarriers
    #   User k occupies subcarriers: k, K+k, 2K+k, ..., (P-1)*K + k
    #   (PDF page 9/11: "P sub-carriers 1, K+1, 2K+1, …, (N-K+1)")
    X = np.zeros(N, dtype=complex)
    subcarrier_indices = user_k + np.arange(P) * K   # K-spaced, offset by k
    X[subcarrier_indices] = precoded

    # Step 3: N-pt IFFT  (PDF: "NxN IFFT")
    return oversample_freq(X, N, L_OS)

# =============================================================================
# TRANSMITTER 4 : DFT-spread OFDMA  (a.k.a. SC-FDMA, LTE Uplink)
# =============================================================================
# Similar to IFDMA but with LOCALIZED subcarrier mapping instead of distributed.
# PDF page 3 lists "DFT-Precoded OFDMA" as a separate GMC flavour.
# PDF page 7: "3GPP LTE has adopted this for UL"
#
# Pipeline: P QAM symbols → P-pt DFT → map to P CONTIGUOUS subcarriers
#           → N-pt IFFT → Add CP
# =============================================================================
def tx_dft_spread_ofdma(symbols_P):
    """
    DFT-spread OFDMA (SC-FDMA) transmitter:
      1) P-pt DFT precoding
      2) Map to P contiguous (localized) subcarriers
      3) N-pt IFFT
    """
    # Step 1: P-point DFT precoding
    precoded = fft(symbols_P) / np.sqrt(P)

    # Step 2: Localized mapping — P contiguous subcarriers starting at 0
    X = np.zeros(N, dtype=complex)
    X[0:P] = precoded

    # Step 3: N-pt IFFT
    return oversample_freq(X, N, L_OS)

# =============================================================================
# Monte Carlo Simulation Loop
# =============================================================================
print(f"Running PAPR simulation: N={N}, K={K}, P={P}, {QAM_ORDER}-QAM")
print(f"  Oversampling factor L={L_OS}, Iterations={NUM_ITER}")
print("  This may take a minute ...")

papr_ofdma     = np.zeros(NUM_ITER)
papr_fdoss     = np.zeros(NUM_ITER)
papr_ifdma     = np.zeros(NUM_ITER)
papr_dft_sofdm = np.zeros(NUM_ITER)

for i in range(NUM_ITER):
    # Generate P random 16-QAM symbols (one user's data block)
    syms = gen_qam_symbols(QAM_ORDER, P)

    # --- Tx 1: Baseline OFDMA (PDF p.4) ---
    sig_ofdma = tx_ofdma(syms)
    papr_ofdma[i] = calc_papr_dB(sig_ofdma)

    # --- Tx 2: F-DOSS (PDF p.8) ---
    sig_fdoss = tx_fdoss(syms, user_k=0)
    papr_fdoss[i] = calc_papr_dB(sig_fdoss)

    # --- Tx 3: IFDMA (PDF p.11) ---
    sig_ifdma = tx_ifdma(syms, user_k=0)
    papr_ifdma[i] = calc_papr_dB(sig_ifdma)

    # --- Tx 4: DFT-spread OFDMA (PDF p.3,7) ---
    sig_dfts = tx_dft_spread_ofdma(syms)
    papr_dft_sofdm[i] = calc_papr_dB(sig_dfts)

    if (i + 1) % 2000 == 0:
        print(f"  ... {i+1}/{NUM_ITER} iterations done")

print("Simulation complete.\n")

# =============================================================================
# CCDF Plotting
# =============================================================================
def plot_ccdf(papr_arrays, labels, colors, styles, title):
    """
    Plot Pr(PAPR > PAPR_0) vs PAPR_0  for each scheme.
    CCDF = 1 - CDF.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for papr_vals, lbl, clr, ls in zip(papr_arrays, labels, colors, styles):
        sorted_papr = np.sort(papr_vals)
        ccdf = 1.0 - np.arange(1, len(sorted_papr) + 1) / len(sorted_papr)
        ax.semilogy(sorted_papr, ccdf, ls, color=clr, linewidth=2.0, label=lbl)

    ax.set_xlabel("PAPR$_0$ (dB)", fontsize=13)
    ax.set_ylabel("Pr(PAPR > PAPR$_0$)  [CCDF]", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, which='both', alpha=0.4)
    ax.set_ylim([1e-4, 1])
    ax.set_xlim([0, 14])
    fig.tight_layout()

    # Save high-res figure
    fig.savefig("papr_ccdf_comparison.png", dpi=200, bbox_inches='tight')
    print("Figure saved to: papr_ccdf_comparison.png")
    plt.show()

# Print summary statistics
print("=" * 60)
print("PAPR Summary (dB)  –  Mean / 99th-percentile")
print("=" * 60)
for name, arr in [("OFDMA (baseline)",     papr_ofdma),
                  ("F-DOSS",               papr_fdoss),
                  ("IFDMA",                papr_ifdma),
                  ("DFT-spread OFDMA",     papr_dft_sofdm)]:
    print(f"  {name:25s}:  mean={np.mean(arr):.2f} dB,  "
          f"99th%={np.percentile(arr,99):.2f} dB")
print("=" * 60)

plot_ccdf(
    [papr_ofdma, papr_fdoss, papr_ifdma, papr_dft_sofdm],
    ["OFDMA (baseline)", "F-DOSS", "IFDMA (distributed)", "DFT-spread OFDMA (localized)"],
    ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"],
    ["-", "-", "--", "-."],
    f"CCDF of PAPR  |  N={N}, P={P}, K={K}, {QAM_ORDER}-QAM, {NUM_ITER} frames"
)
