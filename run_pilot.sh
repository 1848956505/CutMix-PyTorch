#!/usr/bin/env bash
set -euo pipefail

UPDATES="${UPDATES:-10000}"
SEED="${SEED:-20170922}"
AMP_FLAG="${AMP_FLAG:---amp}"

mkdir -p cross_layer/logs

for GROUP in g2 g3 g4; do
  echo
  echo "================================================================"
  echo "Running ${GROUP} | seed=${SEED} | updates=${UPDATES}"
  echo "================================================================"

  python cross_layer/train_pilot.py \
    --group "${GROUP}" \
    --seed "${SEED}" \
    --max-updates "${UPDATES}" \
    --eval-every 1000 \
    --log-every 100 \
    --source-batch 32 \
    --repeats 4 \
    --lr 0.1 \
    --weight-decay 5e-4 \
    --ra-n 3 \
    --fixed-m 5 \
    --role-b-m 9 \
    --m-min 3 \
    --m-max 9 \
    ${AMP_FLAG} \
    2>&1 | tee "cross_layer/logs/${GROUP}_seed${SEED}_u${UPDATES}.log"
done

python cross_layer/collect_pilot.py \
  --seed "${SEED}" \
  --updates "${UPDATES}"
