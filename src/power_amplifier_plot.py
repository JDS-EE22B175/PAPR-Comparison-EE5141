"""
Plot a realistic Power Amplifier (PA) transfer curve.
Shows output power vs input power (dBm), the ideal linear reference,
the actual saturating PA curve, and the 1dB compression point (P1dB).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# 1. PA Model (Saleh / soft-limiter approximation)
# ─────────────────────────────────────────────────────────────────────────────

def pa_output_dBm(P_in_dBm: np.ndarray, gain_dB: float = 22.0,
                  P_sat_dBm: float = 30.0) -> np.ndarray:
    """
    Soft-limiter PA model.
    Pout(dBm) approaches P_sat_dBm asymptotically as Pin increases.
    gain_dB  : small-signal gain
    P_sat_dBm: output saturation power
    """
    P_in_lin  = 10.0 ** (P_in_dBm  / 10.0)   # mW
    P_sat_lin = 10.0 ** (P_sat_dBm / 10.0)    # mW
    G_lin     = 10.0 ** (gain_dB   / 10.0)

    # Rapp model (p=2): smooth transition to saturation
    P_out_lin = (G_lin * P_in_lin) / (1.0 + (G_lin * P_in_lin / P_sat_lin) ** 2) ** 0.5
    return 10.0 * np.log10(P_out_lin)          # back to dBm


def linear_ref_dBm(P_in_dBm: np.ndarray, gain_dB: float = 22.0) -> np.ndarray:
    """Ideal linear amplifier: Pout = Pin + G (in dB domain)."""
    return P_in_dBm + gain_dB


def find_P1dB(gain_dB: float = 22.0, P_sat_dBm: float = 25.0) -> tuple[float, float]:
    """
    Find the 1 dB compression point by scanning.
    Returns (P_in_P1dB, P_out_P1dB) in dBm.
    """
    P_in_scan = np.linspace(-30, P_sat_dBm - gain_dB + 5, 5000)
    P_out_actual = pa_output_dBm(P_in_scan, gain_dB, P_sat_dBm)
    P_out_linear = linear_ref_dBm(P_in_scan, gain_dB)
    compression  = P_out_linear - P_out_actual   # how far below linear
    # P1dB is where compression first reaches 1 dB
    idx = np.argmin(np.abs(compression - 1.0))
    return float(P_in_scan[idx]), float(P_out_actual[idx])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_pa_curve(save_path: str = "outputs/pa_curve.png") -> None:
    """Generates and saves the PA transfer-curve plot."""
    GAIN_DB   = 22.0   # small-signal gain
    P_SAT_DBM = 25.0   # output saturation power (dBm)

    P_in = np.linspace(-5, 10, 1000)    # input power sweep (dBm) — zoomed to compression region
    P_out_actual = pa_output_dBm(P_in, GAIN_DB, P_SAT_DBM)
    P_out_linear = linear_ref_dBm(P_in, GAIN_DB)

    P1dB_in, P1dB_out = find_P1dB(GAIN_DB, P_SAT_DBM)

    fig, ax = plt.subplots(figsize=(9, 6))

    # -- Actual PA curve --
    ax.plot(P_in, P_out_actual, color='#e74c3c', linewidth=2.5,
            label='PA Output (Rapp model, $p=2$)')

    # -- Ideal linear reference --
    ax.plot(P_in, P_out_linear, color='#3498db', linewidth=2.0,
            linestyle='--', label=f'Ideal Linear ($G = {GAIN_DB:.0f}$ dB)')

    # -- Saturation ceiling --
    ax.axhline(P_SAT_DBM, color='gray', linewidth=1.2, linestyle=':',
               label=f'$P_{{sat}} = {P_SAT_DBM:.0f}$ dBm')

    # -- 1dB Compression Point annotation --
    ax.annotate(
        f'$P_{{1dB}}$\n({P1dB_in:.1f}, {P1dB_out:.1f}) dBm',
        xy=(P1dB_in, P1dB_out),
        xytext=(P1dB_in - 5, P1dB_out),
        fontsize=11, color='#2c3e50',
        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f', alpha=0.85),
    )
    ax.plot(P1dB_in, P1dB_out, 'ko', markersize=8, zorder=5)

    # -- 1dB drop bracket --
    ax.annotate('', xy=(P1dB_in, P1dB_out),
                xytext=(P1dB_in, P1dB_out + 1.0),
                arrowprops=dict(arrowstyle='<->', color='#27ae60', lw=1.5))
    ax.text(P1dB_in + 0.4, P1dB_out + 0.5, '1 dB', fontsize=10,
            color='#27ae60', va='center')

    # -- Shading: linear region vs compression region --
    ax.axvspan(P_in[0], P1dB_in, alpha=0.06, color='#2ecc71')
    ax.axvspan(P1dB_in, P_in[-1], alpha=0.06, color='#e74c3c')

    # -- Region Labels inside the plot --
    ax.text((P_in[0] + P1dB_in)/2, P_in[0] + GAIN_DB + 1, 'Linear Region', 
            ha='center', va='center', fontsize=11, color='#27ae60', fontweight='bold', alpha=0.8)
    ax.text((P1dB_in + P_in[-1])/2, P_in[0] + GAIN_DB + 1, 'Compression Region', 
            ha='center', va='center', fontsize=11, color='#c0392b', fontweight='bold', alpha=0.8)

    ax.set_xlabel('Input Power $P_{in}$ (dBm)', fontsize=13)
    ax.set_ylabel('Output Power $P_{out}$ (dBm)', fontsize=13)
    ax.set_title('PA Transfer Curve & Efficiency', fontsize=14, fontweight='bold')
    
    # -- Efficiency Overlay (Right Y-Axis) --
    ax2 = ax.twinx()
    # Simplified efficiency model: peaks at saturation
    P_out_lin = 10.0 ** (P_out_actual / 10.0)
    P_sat_lin = 10.0 ** (P_SAT_DBM / 10.0)
    efficiency = 60.0 * np.sqrt(P_out_lin / P_sat_lin) # Example: 60% max efficiency
    
    line_eff, = ax2.plot(P_in, efficiency, color='#8e44ad', linewidth=2.0, 
                         linestyle='-.', label='Efficiency (%)')
    ax2.set_ylabel('Efficiency (%)', fontsize=13, color='#8e44ad')
    ax2.tick_params(axis='y', labelcolor='#8e44ad')
    ax2.set_ylim(0, 100)

    # Combine legends
    lines_ax, labels_ax = ax.get_legend_handles_labels()
    ax.legend(lines_ax + [line_eff], labels_ax + ['Efficiency (%)'], 
              fontsize=10, loc='upper left', framealpha=0.9)
    
    ax.grid(True, alpha=0.35, linestyle='--')
    ax.set_xlim(P_in[0], P_in[-1])
    ax.set_ylim(P_in[0] + GAIN_DB - 2, P_SAT_DBM + 3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    print("Generating PA curve plot...")
    plot_pa_curve("outputs/pa_curve.png")
