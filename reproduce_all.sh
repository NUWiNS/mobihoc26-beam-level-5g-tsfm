#!/bin/bash
# Reproduce all figures, tables and Section-4 numbers of the paper from the released
# per-beam results (CPU only, ~5 min). GPU experiments are listed in README.md.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -f data/gotsf/train/DLThpVol_train_0w-5w.csv ] || ./data/download_data.sh
mkdir -p figures/out logs
echo "[1/5] Section 4 characterization numbers";   $PY audit/verify_char.py        | tee logs/verify_char.log
$PY audit/verify_char_extra.py | tee logs/verify_char_extra.log
echo "[2/5] Tables 5-7, statistics (Friedman/Nemenyi, bootstrap, Wilcoxon)"; $PY analysis/analysis.py > logs/analysis.log; echo "  -> logs/analysis.log"
echo "[3/5] Graph WaveNet summary";                $PY experiments/analyze_gwn.py | tee logs/gwn.log
echo "[4/5] Figures";
$PY analysis/deep_measurement.py > logs/deep_measurement.log        # Fig. 1, Fig. 5
$PY figures/generate_figures.py  > logs/generate_figures.log        # Fig. 2
$PY figures/make_figs.py data/gotsf/train results figures/out       # Fig. 3, 4, 8
$PY figures/make_fig7.py results/per_beam_results.csv figures/out/fig7_covariate_effect   # Fig. 6
$PY figures/make_rankfig.py results figures/out                     # Fig. 7
echo "[5/5] Done. Figures in figures/out/, logs in logs/"
