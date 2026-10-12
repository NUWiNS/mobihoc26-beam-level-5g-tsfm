# Artifact for: Measuring the Predictability of Beam-Level 5G Traffic: How Far Can Zero-Shot Foundation Models Go?
ACM MobiHoc 2026

---

## Overview

This artifact accompanies our MobiHoc '26 paper, a measurement study of beam-level 5G traffic and a benchmark of 16 forecasting configurations on it: zero-shot time-series foundation models (Chronos-2, Chronos-Bolt, Moirai, TimesFM), a RAN-specific foundation model (TimeRAN), trained deep-learning baselines (PatchTST, TiDE, N-HiTS, DLinear, N-BEATSx; with and without covariates) and ARIMA, plus two reference baselines (Graph WaveNet, univariate Chronos-2 cross-learning).

The artifact includes:
- Per-beam results for every model, test week and dataset (`results/`)
- Scripts to reproduce all figures, tables and characterization numbers from those results (CPU only)
- The full experiment code to regenerate the results (GPU)
- Audit scripts that recompute every number in Section 4 directly from the raw data

The datasets are public and are downloaded from their original sources (not redistributed here).

---

## Directory Structure

```text
.
├── analysis/          # Statistics (bootstrap, Friedman/Nemenyi, Wilcoxon) and beam-level characterization
├── audit/             # Recompute every Section-4 number from the raw data
├── data/              # download_data.sh -> data/gotsf/ and data/irish/
├── experiments/       # Non-oracle covariates, beam-sleep sweep, Graph WaveNet, cross-learning ablation
├── figures/           # Figure scripts; output in figures/out/
├── results/           # Per-beam results used in the paper
├── benchmark.py       # Main benchmark (all zero-shot / trained / ARIMA models, metrics)
├── run_timeran.py     # TimeRAN fine-tuning / linear probing (+ setup_timeran.sh)
├── paths.py           # File locations (override with GOTSF_DIR, IRISH_DIR, RESULTS_DIR, FIG_DIR)
├── reproduce_all.sh   # Master script: all figures, tables and Section-4 numbers
└── README.md          # This file
```

---

## System Requirements

- OS: Linux or macOS
- Python: 3.10
- Reproducing figures/tables (`reproduce_all.sh`): CPU only, ~2 minutes, ~250 MB disk for the data
- Re-running the experiments: one NVIDIA GPU (we used an A100 80 GB)

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Reproduce All Results

```bash
./reproduce_all.sh
```

This downloads the datasets (if needed) and writes figures to `figures/out/` and logs to `logs/`:

| Paper | Output |
|---|---|
| Fig. 1 (BS / cell / beam disaggregation) | `figures/out/fig_cell_vs_beam.pdf` |
| Fig. 2 (zero-fraction distribution) | `figures/out/fig1_sparsity_distribution.pdf` |
| Fig. 3 (covariate–target correlation) | `figures/out/fig2_covariate_correlation.pdf` |
| Fig. 4 (temporal patterns) | `figures/out/fig3_temporal_patterns.pdf` |
| Fig. 5 (spatial correlations) | `figures/out/fig_spatial_correlations.pdf` |
| Fig. 6 (covariate effect) | `figures/out/fig7_covariate_effect.pdf` |
| Fig. 7 (Friedman mean ranks) | `figures/out/fig6_nemenyi_cd.pdf` |
| Fig. 8 (MASE by sparsity bin) | `figures/out/fig4_mase_by_sparsity.pdf` |
| Tables 5–7, statistical tests, Irish results | `logs/analysis.log` |
| Graph WaveNet († rows) | `logs/gwn.log` |
| Section 4 numbers (ACF, correlations, co-sparsity, shift, Irish properties) | `logs/verify_char*.log` (STL shares: `python audit/verify_stl.py`) |

## Re-running the Experiments (GPU)

| Paper item | Command | Output |
|---|---|---|
| Table 5/6/7 models (GOTSF) | `python benchmark.py --num-bs 30 --max-steps 10000 --test-weeks 5w-6w 10w-11w` | `results/per_beam_results.csv` |
| Irish 5G | `python benchmark.py --dataset irish --max-steps 10000` | `results/per_beam_results_irish.csv` |
| TimeRAN (GOTSF BS 0–4, Irish) | `./setup_timeran.sh`, then `python run_timeran.py --test-weeks 5w-6w 10w-11w --mode {full_finetuning,linear_probing}` and `--dataset irish` | appended to the files above |
| Table 4 (non-oracle covariates), context ablation | `python experiments/nonoracle_experiment.py && python experiments/nonoracle_part2.py` | `results/nonoracle_results.csv` |
| Graph WaveNet | `python experiments/run_gwn.py --seed {42,0,1,2,3}` | `results/gwn/` (tuning runs: `results/gwn_tuning/`) |
| Cross-learning without covariates | `python experiments/xl_uni.py` | `results/audit/xl_uni.csv` |
| Beam-sleep evaluation (Sec. 7.2) | `python experiments/sleep_sweep.py` | `results/sleep_sweep.csv` |

Notes: covariate-augmented results use oracle (ground-truth) future covariates; NeuralForecast baselines are one global model over all beams (seed 42); Chronos2-XL uses the same covariates as Chronos2-Cov plus cross-learning within each base station. With `transformers` 4.33 and `chronos-forecasting` 2.2.2, Chronos-2 needs the attention-config shim included in `experiments/xl_uni.py` and `experiments/sleep_sweep.py`; it does not change predictions.

---

## Results Files

- `results/per_beam_results.csv` — one row per (beam, model, test week): MAE, RMSE, MASE, NMAE, PICP_80, CRPS. `test_week` is `5w-6w` (Week 6) or `10w-11w` (Week 11). TimeRAN rows cover BS 0–4 (480 beams).
- `results/per_beam_results_irish.csv` — same for the 81 Irish 5G sessions.
- `results/beam_metadata*.csv` — per-beam training zero fraction. Paper sparsity bins: `pd.cut(zero_frac, [0, .1, .3, .5, .7], include_lowest=True)`.
- `results/nonoracle_results.csv`, `results/gwn/`, `results/gwn_tuning/`, `results/sleep_sweep.csv`, `results/audit/`.

---

## Datasets

Downloaded by `data/download_data.sh` from the original sources. Please cite them if you use them:

- **GOTSF** (MIT), Huawei NETOP — https://huggingface.co/datasets/netop/gotsf-ds — L.-A. Fechete et al., *Goal-Oriented Time-Series Forecasting: Foundation Framework Design*, AAAI 2026.
- **Irish 5G** (GPL-3.0), University College Cork — https://github.com/uccmisl/5Gdataset — D. Raca et al., *Beyond Throughput, the Next Generation: A 5G Dataset with Channel and Context Metrics*, ACM MMSys 2020.

---

## Citation

If you use this artifact, please cite:

> Eduardo Baena, Phuc Dinh, Sizhe Wang, and Dimitrios Koutsonikolas. 2026. Measuring the Predictability of Beam-Level 5G Traffic: How Far Can Zero-Shot Foundation Models Go?. In *The Twenty-seventh International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc '26), November 23–26, 2026, Tokyo, Japan.* ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3842721.3849689

```bibtex
@inproceedings{baena2026beam,
  author    = {Baena, Eduardo and Dinh, Phuc and Wang, Sizhe and Koutsonikolas, Dimitrios},
  title     = {Measuring the Predictability of Beam-Level {5G} Traffic: How Far Can Zero-Shot Foundation Models Go?},
  booktitle = {Proceedings of the Twenty-seventh International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc '26)},
  year      = {2026},
  month     = nov,
  address   = {Tokyo, Japan},
  publisher = {ACM},
  numpages  = {10},
  doi       = {10.1145/3842721.3849689}
}
```

---

## License

Code: MIT (see `LICENSE`). The paper is published under CC BY-NC-ND 4.0. Datasets keep their original licenses (above).

## Acknowledgments

This work was supported in part by NSF grant CNS-2340283.
