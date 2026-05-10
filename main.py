"""
Main script to run the PAPR simulation.
Will do a Monte Carlo simulation for OFDMA, F-DOSS, IFDMA, and SC-FDMA.
"""

import argparse
import time
import os
import numpy as np

from src import config
from src.config import SCHEME_NAMES, SCHEME_FUNCS
import src.transmitters  # noqa: F401 - need this to register schemes
from src.utils import gen_qam_symbols, calc_papr_dB
from src.plotting import plot_single_qam, plot_combined
from src.export_excel import export_results

def run_monte_carlo(qam_order: int, num_iter: int) -> np.ndarray:
    """Runs the simulation for a specific QAM order."""
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
    parser.add_argument("--quick", action="store_true", help="Run fewer iterations for testing")
    parser.add_argument("--extended", action="store_true",
                        help="Run extended analysis (N-sweep, K-sweep, clipping, theory)")
    parser.add_argument("--equivalence", action="store_true",
                        help="Run F-DOSS/IFDMA equivalence verification")
    args = parser.parse_args()

    # Create outputs folder if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    num_iter = 1000 if args.quick else config.NUM_ITER
    np.random.seed(config.RANDOM_SEED)

    print("========================================")
    print(" PAPR Comparison Simulation")
    print(f" N={config.N}, K={config.K}, P={config.P}, L={config.L_OS}, Iterations={num_iter}")
    print("========================================")

    # Run the simulations
    all_results = {}
    for qam in config.QAM_ORDERS:
        print(f"\n--- {qam}-QAM ---")
        t0 = time.time()
        papr_db = run_monte_carlo(qam, num_iter)
        elapsed = time.time() - t0
        all_results[qam] = papr_db

        print(f"  Done in {elapsed:.1f}s")
        for s, name in enumerate(SCHEME_NAMES):
            mean_val = np.mean(papr_db[s])
            p99 = np.percentile(papr_db[s], 99)
            print(f"  {name:15s}: mean={mean_val:.2f}dB, 99%={p99:.2f}dB")

    # Generate the plots
    print("\nGenerating plots in outputs/ ...")
    for qam in config.QAM_ORDERS:
        plot_single_qam(all_results[qam], qam, os.path.join("outputs", f"papr_ccdf_{qam}qam.png"))
    plot_combined(all_results, config.QAM_ORDERS, os.path.join("outputs", "papr_ccdf_combined.png"))

    # Export to excel
    print("Exporting data to Excel ...")
    
    # Values from the other buggy code to compare against
    other_code = {
        4:  {0.01: [9.70, 9.76, 9.68, 9.68], 0.001: [10.62, 10.65, 10.56, 10.56]},
        16: {0.01: [9.76, 9.74, 9.68, 9.68], 0.001: [10.62, 10.65, 10.57, 10.57]},
        64: {0.01: [9.70, 9.73, 9.65, 9.65], 0.001: [10.60, 10.68, 10.60, 10.60]},
    }
    export_results(all_results, os.path.join("outputs", "papr_results.xlsx"), other_code_results=other_code)

    # ── F-DOSS / IFDMA Equivalence Verification ──
    if args.equivalence:
        print("\n========================================")
        print(" F-DOSS / IFDMA Equivalence Verification")
        print("========================================")
        from src.equivalence import verify_equivalence, plot_equivalence
        
        result = verify_equivalence(qam_order=16, num_trials=100)
        print(f"  Normalized shape error (mean): {result['mean_shape_error']:.2e}")
        print(f"  Normalized shape error (worst): {result['worst_shape_error']:.2e}")
        print(f"  PAPR difference (mean):  {result['mean_papr_diff']:.2e} dB")
        print(f"  PAPR difference (worst): {result['worst_papr_diff']:.2e} dB")
        print(f"  Waveforms identical: {result['shapes_match']}")
        print(f"  PAPR identical:      {result['papr_match']}")
        
        plot_equivalence(16, os.path.join("outputs", "fdoss_ifdma_equivalence.png"))

    # ── Extended Analysis ──
    if args.extended:
        from src.extended_analysis import (
            papr_vs_N_sweep, plot_papr_vs_N,
            papr_vs_K_sweep, plot_papr_vs_K,
            plot_ccdf_with_theory,
            clipping_analysis, plot_clipping_tradeoff, plot_clipping_ccdf,
        )
        
        ext_iter = 2000 if args.quick else 5000

        # 1. PAPR vs N sweep
        print("\n========================================")
        print(" Extended: PAPR vs FFT Size (N sweep)")
        print("========================================")
        t0 = time.time()
        n_results = papr_vs_N_sweep(
            N_values=[32, 64, 128, 256, 512, 1024],
            K_fixed=4, qam_order=16, num_iter=ext_iter)
        plot_papr_vs_N(n_results, os.path.join("outputs", "papr_vs_N.png"))
        print(f"  N-sweep done in {time.time() - t0:.1f}s")

        # 2. PAPR vs K sweep
        print("\n========================================")
        print(" Extended: PAPR vs Users (K sweep)")
        print("========================================")
        t0 = time.time()
        k_results = papr_vs_K_sweep(
            N_fixed=256, qam_order=16, num_iter=ext_iter)
        plot_papr_vs_K(k_results, 256, os.path.join("outputs", "papr_vs_K.png"))
        print(f"  K-sweep done in {time.time() - t0:.1f}s")

        # 3. Theoretical OFDMA bound overlay
        print("\n========================================")
        print(" Extended: Theoretical CCDF Overlay")
        print("========================================")
        for qam in config.QAM_ORDERS:
            plot_ccdf_with_theory(
                all_results[qam], qam, config.N,
                os.path.join("outputs", f"papr_ccdf_theory_{qam}qam.png"))

        # 4. Clipping & EVM tradeoff
        print("\n========================================")
        print(" Extended: Clipping & EVM Tradeoff")
        print("========================================")
        t0 = time.time()
        clip_res = clipping_analysis(
            qam_order=16, N_val=256, K_val=4,
            num_iter=ext_iter // 2)
        plot_clipping_tradeoff(clip_res, os.path.join("outputs", "clipping_tradeoff.png"))
        plot_clipping_ccdf(
            qam_order=16, num_iter=ext_iter,
            save_path=os.path.join("outputs", "clipping_ccdf.png"))
        print(f"  Clipping analysis done in {time.time() - t0:.1f}s")

    print("\nAll done!")

if __name__ == "__main__":
    main()
