# PAPR Comparison: OFDMA vs GMC Uplink Schemes

> **EE5141 – Wireless & Cellular Communications, IIT Madras (Semester 8)**
> Mini Project

## Problem Statement

Compare the Peak-to-Average Power Ratio (PAPR) of four multicarrier modulation schemes used in cellular uplink:

| # | Scheme | Subcarrier Mapping | DFT Precoding? |
|---|--------|--------------------|----------------|
| 1 | **OFDMA** (baseline) | Localized | No |
| 2 | **F-DOSS** | Distributed (implicit) | No (time-domain repetition) |
| 3 | **IFDMA** | Distributed (interleaved) | Yes (P-pt DFT) |
| 4 | **DFT-spread OFDMA** (SC-FDMA) | Localized | Yes (P-pt DFT) |

## Key Results

At **CCDF = 1% (99th percentile)**, with N=256, P=64, K=4:

| Scheme | QPSK | 16-QAM | 64-QAM |
|--------|------|--------|--------|
| OFDMA | 9.82 dB | 9.82 dB | 9.79 dB |
| F-DOSS | 6.97 dB | 7.80 dB | 7.98 dB |
| IFDMA | 6.97 dB | 7.80 dB | 7.98 dB |
| DFT-s-OFDMA | 6.98 dB | 7.80 dB | 7.92 dB |

**All three GMC schemes achieve ~2-3 dB PAPR reduction over OFDMA.**

## References

1. K. Giridhar, "Generalised Multi-Carrier (GMC) ++", IIT Madras (course slides)
2. H.G. Myung, J. Lim, D.J. Goodman, "Single Carrier FDMA for Uplink Wireless Transmission", IEEE Vehicular Technology Magazine, Sep 2006
3. Chang & Chen, IEEE Communications Letters, Nov 2000 (F-DOSS)

## Quick Start

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run simulation (~20 seconds)
python papr_simulation.py
```

## Output Files

| File | Description |
|------|-------------|
| `papr_ccdf_4qam.png` | CCDF plot for QPSK |
| `papr_ccdf_16qam.png` | CCDF plot for 16-QAM |
| `papr_ccdf_64qam.png` | CCDF plot for 64-QAM |
| `papr_ccdf_combined.png` | Side-by-side comparison |
| `papr_results.xlsx` | Summary statistics + raw data |

## Project Structure

```
├── main.py            # Entry point
├── src/               # Core logic (package)
│   ├── config.py
│   ├── transmitters.py
│   ├── utils.py
│   ├── plotting.py
│   └── export_excel.py
├── outputs/           # Generated results (Excel, PNGs)
├── legacy/            # Original scripts
├── requirements.txt
├── README.md
└── .gitignore
```

## License

Academic use — EE5141 Mini Project, IIT Madras.
