"""
Main entry point — orchestrates Monte Carlo simulation, plotting, and export.

Usage:
    python main.py                  # full run (3 QAM orders × 10k iterations)
    python main.py --quick          # quick test (1k iterations)

All outputs are saved to the current directory.
"""

import argparse
import time
import numpy as np

import config
from config import SCHEME_NAMES, SCHEME_FUNCS
import transmitters          # noqa: F401 — registers schemes into config
from utils import gen_qam_symbols, calc_papr_dB
from plotting import plot_single_qam, plot_combined
from export_excel import export_results


def run_monte_carlo(qam_order: int, num_iter: int) -> np.ndarray:
    """
    Run Monte Carlo PAPR simulation for one QAM modulation order.

    Returns
    -------
    papr_db : ndarray, shape (num_schemes, num_iter)
    """
    num_schemes = len(SCHEME_NAMES)
    papr_db = np.zeros((num_schemes, num_iter))

    for i in range(num_iter):
        syms = gen_qam_symbols(qam_order, config.P)
        for s, func in enumerate(SCHEME_FUNCS):
            sig = func(syms)
            papr_db[s, i] = calc_papr_dB(sig)

    return papr_db


def main():
    parser = argparse.ArgumentParser(description="PAPR CCDF Simulation")
    parser.add_argument("--quick", action="store_true",
                        help="Run with 1000 iterations instead of 10000")
    args = parser.parse_args()

    num_iter = 1000 if args.quick else config.NUM_ITER
    np.random.seed(config.RANDOM_SEED)

    print("=" * 70)
    print("  PAPR Comparison Simulation")
    print(f"  N={config.N}, K={config.K}, P={config.P},"
          f" L={config.L_OS}, Iterations={num_iter}")
    print(f"  QAM orders: {config.QAM_ORDERS}")
    print(f"  Schemes: {SCHEME_NAMES}")
    print("=" * 70)

    # ──────────── Monte Carlo ───────────────────────────────────────────
    all_results = {}
    for qam in config.QAM_ORDERS:
        print(f"\n>>> {qam}-QAM ...")
        t0 = time.time()
        papr_db = run_monte_carlo(qam, num_iter)
        elapsed = time.time() - t0
        all_results[qam] = papr_db

        print(f"    Done in {elapsed:.1f}s")
        for s, name in enumerate(SCHEME_NAMES):
            m = np.mean(papr_db[s])
            p99 = np.percentile(papr_db[s], 99)
            p999 = np.percentile(papr_db[s], 99.9)
            print(f"    {name:18s}:  mean={m:.2f},  "
                  f"99th%={p99:.2f},  99.9th%={p999:.2f} dB")

    # ──────────── Plots ─────────────────────────────────────────────────
    print("\nGenerating plots ...")
    for qam in config.QAM_ORDERS:
        plot_single_qam(all_results[qam], qam, f"papr_ccdf_{qam}qam.png")
    plot_combined(all_results, config.QAM_ORDERS, "papr_ccdf_combined.png")

    # ──────────── Excel ─────────────────────────────────────────────────
    print("\nExporting Excel ...")
    # "Other code" results provided by the user for comparison
    other_code = {
        4:  {0.01: [9.70, 9.76, 9.68, 9.68],
             0.001: [10.62, 10.65, 10.56, 10.56]},
        16: {0.01: [9.76, 9.74, 9.68, 9.68],
             0.001: [10.62, 10.65, 10.57, 10.57]},
        64: {0.01: [9.70, 9.73, 9.65, 9.65],
             0.001: [10.60, 10.68, 10.60, 10.60]},
    }
    export_results(all_results, "papr_results.xlsx",
                   other_code_results=other_code)

    # ──────────── Done ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  All outputs generated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
