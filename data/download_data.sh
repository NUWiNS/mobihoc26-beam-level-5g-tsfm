#!/bin/bash
# Download the two public datasets used in the paper into data/ (nothing is redistributed here).
#   GOTSF  (MIT)      https://huggingface.co/datasets/netop/gotsf-ds      -> data/gotsf/{train,test}/
#   Irish 5G (GPL-3.0) https://github.com/uccmisl/5Gdataset                 -> data/irish/
set -e
cd "$(dirname "$0")"
HF=https://huggingface.co/datasets/netop/gotsf-ds/resolve/main/data
mkdir -p gotsf/train gotsf/test
for kpi in DLThpVol DLThpTime DLPRB MR_number; do
  curl -fL -o "gotsf/train/${kpi}_train_0w-5w.csv" "$HF/train/${kpi}_train_0w-5w.csv"
  for wk in 5w-6w 10w-11w; do
    curl -fL -o "gotsf/test/${kpi}_test_${wk}.csv" "$HF/test/${kpi}_test_${wk}.csv"
  done
done
mkdir -p irish
curl -fL -o irish/5G-production-dataset.zip https://github.com/uccmisl/5Gdataset/raw/master/5G-production-dataset.zip
(cd irish && unzip -q -o 5G-production-dataset.zip && rm -rf 5G-production-dataset.zip __MACOSX)
echo "Done: data/gotsf and data/irish"
