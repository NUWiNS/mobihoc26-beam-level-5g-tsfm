# Measuring the Predictability of Beam-Level 5G Traffic: How Far Can Zero-Shot Foundation Models Go?

Code, per-beam results, and reproducibility scripts for the MobiHoc '26 paper

> Eduardo Baena, Phuc Dinh, Sizhe Wang, and Dimitrios Koutsonikolas. 2026. *Measuring the Predictability of Beam-Level 5G Traffic: How Far Can Zero-Shot Foundation Models Go?* In Proc. of ACM MobiHoc '26, Tokyo, Japan. https://doi.org/10.1145/3842721.3849689

We characterize 2,880 operator-side beam-level series (GOTSF) and 81 client-side sessions (Irish 5G), and benchmark 16 forecasting configurations: zero-shot time-series foundation models (Chronos-2, Chronos-Bolt, Moirai, TimesFM), a RAN-specific foundation model (TimeRAN), trained deep-learning baselines (PatchTST, TiDE, N-HiTS, DLinear, N-BEATSx; with and without covariates), and ARIMA. Additional references: Graph WaveNet (trained ST-GNN) and univariate Chronos-2 cross-learning.

## Repository layout

| Path | Content |
|---|---|
| `paths.py` | All file locations (override with env vars `GOTSF_DIR`, `IRISH_DIR`, `RESULTS_DIR`, `FIG_DIR`) |
| `benchmark.py` | Main benchmark: data loading, all zero-shot / trained / ARIMA models, metrics (MASE, CRPS, PICP) → `results/per_beam_results*.csv` |
| `run_timeran.py`, `setup_timeran.sh` | TimeRAN (MOMENT-1-small backbone) fine-tuning / linear probing, BS 0–4 |
| `analysis/analysis.py` | Aggregate tables, bootstrap CIs, Friedman/Nemenyi, Wilcoxon, sparsity-stratified results |
| `analysis/deep_measurement*.py` | Beam-level characterization (cell vs. beam, spatial correlations, co-sparsity, Irish properties) |
| `experiments/nonoracle_*.py` | Non-oracle covariate pipelines and context-length ablation (289-beam stratified subset) |
| `experiments/sleep_policy.py`, `experiments/sleep_sweep.py` | Decision-level beam-sleep evaluation (interval-gated policy, threshold sweep) |
| `experiments/run_gwn.py`, `experiments/analyze_gwn.py` | Graph WaveNet baseline (5 seeds; hyperparameters selected on validation) |
| `experiments/xl_uni.py` | Chronos-2 cross-learning without covariates |
| `figures/` | Scripts that produce the paper figures |
| `audit/` | Scripts that recompute every characterization number in §4 directly from the raw data |
| `results/` | Per-beam results used in the paper (see below) |

## Setup

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./data/download_data.sh          # downloads GOTSF (Hugging Face) and Irish 5G (GitHub) into data/
./setup_timeran.sh               # only needed for TimeRAN
```

Experiments were run on one NVIDIA A100 (80 GB). Characterization, audit and figure scripts run on a laptop CPU.
Note: with `transformers` 4.33 and `chronos-forecasting` 2.2.2, Chronos-2 needs the one-line attention-config shim included in `experiments/xl_uni.py` and `experiments/sleep_sweep.py`; it does not change predictions (it reproduces the paper's MASE exactly).

## Reproducing the paper

| Paper item | Command | Output |
|---|---|---|
| Table 5 (GOTSF MASE), Table 6 (shift), Table 7 (sparsity), Irish results | `python benchmark.py --num-bs 30 --max-steps 10000 --test-weeks 5w-6w 10w-11w` and `python benchmark.py --dataset irish --max-steps 10000`, then `python analysis/analysis.py` | `results/per_beam_results*.csv` |
| TimeRAN rows | `python run_timeran.py --test-weeks 5w-6w 10w-11w --mode {full_finetuning,linear_probing}` (GOTSF, BS 0–4) and `--dataset irish` | appended to `results/per_beam_results*.csv` |
| Table 4 (non-oracle covariates), context ablation | `python experiments/nonoracle_experiment.py && python experiments/nonoracle_part2.py` | `results/nonoracle_results.csv` |
| Graph WaveNet († rows) | `python experiments/run_gwn.py --seed {42,0,1,2,3}` then `python experiments/analyze_gwn.py` | `results/gwn/` |
| Cross-learning without covariates | `python experiments/xl_uni.py` | `results/audit/xl_uni.csv` |
| Beam-sleep evaluation (§7.2) | `python experiments/sleep_sweep.py` | `results/sleep_sweep.csv` |
| §4 characterization numbers | `python audit/verify_char.py`, `python audit/verify_char_extra.py`, `python audit/verify_stl.py` | printed; STL shares in `results/audit/stl_shares.csv` |
| Fig. 1 (BS/cell/beam), Fig. 5 (spatial) | `python analysis/deep_measurement.py` | `figures/out/` |
| Fig. 2 (sparsity distribution) | `python figures/generate_figures.py` | `figures/out/` |
| Fig. 3, 4, 8 | `python figures/make_figs.py data/gotsf/train results figures/out` | `figures/out/` |
| Fig. 6 (covariate effect) | `python figures/make_fig7.py results/per_beam_results.csv figures/out/fig7_covariate_effect` | `figures/out/` |
| Fig. 7 (Friedman ranks) | `python figures/make_rankfig.py results figures/out` | `figures/out/` |

## Results files

- `results/per_beam_results.csv` — one row per (beam, model, test week) with MAE, RMSE, MASE, NMAE, PICP_80, CRPS; `test_week` is `5w-6w` (Week 6) or `10w-11w` (Week 11). TimeRAN rows cover BS 0–4 (480 beams).
- `results/per_beam_results_irish.csv` — same for the 81 Irish 5G sessions.
- `results/beam_metadata*.csv` — per-beam training zero fraction; paper sparsity bins are `pd.cut(zero_frac, [0, .1, .3, .5, .7], include_lowest=True)`.
- `results/nonoracle_results.csv`, `results/gwn/`, `results/gwn_tuning/`, `results/sleep_sweep.csv`, `results/audit/`.

Notes on the setup (also stated in the paper): covariate-augmented results use oracle (ground-truth) future covariates; NeuralForecast baselines are one global model over all beams with seed 42; Chronos2-XL uses the same covariates as Chronos2-Cov plus cross-learning within each base station.

## Data

The datasets are **not** redistributed; `data/download_data.sh` fetches them from the original sources. Please cite them:

- **GOTSF** (MIT license), Huawei NETOP: https://huggingface.co/datasets/netop/gotsf-ds — L.-A. Fechete et al., *Goal-Oriented Time-Series Forecasting: Foundation Framework Design*, AAAI 2026.
- **Irish 5G** (GPL-3.0), University College Cork: https://github.com/uccmisl/5Gdataset — D. Raca et al., *Beyond Throughput, the Next Generation: A 5G Dataset with Channel and Context Metrics*, ACM MMSys 2020.

## Citation

```bibtex
@inproceedings{baena2026beam,
  author    = {Baena, Eduardo and Dinh, Phuc and Wang, Sizhe and Koutsonikolas, Dimitrios},
  title     = {Measuring the Predictability of Beam-Level {5G} Traffic: How Far Can Zero-Shot Foundation Models Go?},
  booktitle = {Proc. of ACM MobiHoc},
  year      = {2026},
  doi       = {10.1145/3842721.3849689}
}
```

This work was supported in part by NSF grant CNS-2340283. Code is released under the MIT license (see `LICENSE`).
