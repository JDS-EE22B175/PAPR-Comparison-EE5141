"""
Script to generate a side-by-side scatter plot of 16-QAM constellations.
Left: Ideal received symbols (no clipping)
Right: Smeared symbols (heavy clipping)
To visually demonstrate the EVM penalty.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.fft import fft

from .utils import gen_qam_symbols
from .extended_analysis import clip_signal
from .config import N, P, L_OS
from .transmitters import tx_ofdma


# ─────────────────────────────────────────────────────────────────────────────
# 1. Transmitter - Channel - Receiver Chain
# ─────────────────────────────────────────────────────────────────────────────

def tx_rx_chain(syms: np.ndarray, cr: float = None) -> np.ndarray:
    """
    Passes symbols through OFDMA Tx, applies optional clipping, and runs Rx.
    
    Args:
        syms: Input QAM symbols.
        cr: Clipping ratio (CR = A_clip / sigma). If None or >= 50, no clipping.
        
    Returns:
        rx_syms: The recovered symbols at the receiver.
    """
    # 1. Transmitter
    sig = tx_ofdma(syms)
    
    # 2. Channel (Clipping)
    if cr is not None and cr < 50:
        sig = clip_signal(sig, cr)
        
    # 3. Receiver
    # The signal is oversampled by L_OS. The N-point data is spread across N*L_OS.
    # Specifically, oversample_freq puts the first N/2 bins at the start, 
    # and the last N/2 at the end of the N*L_OS array.
    N_os = N * L_OS
    S_rx = fft(sig) * np.sqrt(N_os / N)  # Reverse the scaling from oversampling
    
    # Extract the P subcarriers which are at indices 0 to P-1 for OFDMA
    rx_syms = S_rx[:P]
    return rx_syms


# ─────────────────────────────────────────────────────────────────────────────
# 2. Constellation Plot Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_constellation_plot() -> None:
    """
    Generates and saves the 16-QAM EVM visual demonstration plot.
    """
    np.random.seed(42)
    
    all_clean = []
    all_smeared = []
    
    # Run multiple frames to build up a nice "cloud" of points
    num_frames = 200
    for _ in range(num_frames):
        syms = gen_qam_symbols(16, P)
        rx_clean = tx_rx_chain(syms, cr=100)      # No clipping
        rx_smeared = tx_rx_chain(syms, cr=1.2)    # Heavy clipping (CR=1.2)
        
        all_clean.append(rx_clean)
        all_smeared.append(rx_smeared)
        
    all_clean = np.concatenate(all_clean)
    all_smeared = np.concatenate(all_smeared)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Plot: Smeared (EVM distortion)
    ax.scatter(all_smeared.real, all_smeared.imag, s=15, c='#e74c3c', alpha=0.5, edgecolor='none')
    ax.set_title('OFDMA Constellation Distortion (CR = 1.2)', fontsize=14, fontweight='bold')
    
    # Add an inset box explaining EVM
    evm = np.sqrt(np.mean(np.abs(all_smeared - all_clean)**2) / np.mean(np.abs(all_clean)**2)) * 100
    ax.text(0.95, 0.05, f"EVM ≈ {evm:.1f}%\n(Heavy In-Band Distortion)", 
                 transform=ax.transAxes, fontsize=12, fontweight='bold',
                 ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_aspect('equal', 'box')
    # 16-QAM symbols (normalized to power=1) max out around +/- 1.34
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('In-Phase (I)', fontsize=13)
    ax.set_ylabel('Quadrature (Q)', fontsize=13)
        
    fig.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    save_path = os.path.join('outputs', 'clipping_constellation.png')
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    print("Generating constellation plot...")
    generate_constellation_plot()
