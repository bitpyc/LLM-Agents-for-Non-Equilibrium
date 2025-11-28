#!/bin/bash
mkdir -p logs

models=(
  "deepseek-v3"
)

Nt=400

params=(
  "0.0 0.8"
)

ratio=(
  "0.1"
)
for model in "${models[@]}"; do
  for prm in "${params[@]}"; do
    for r in "${ratio[@]}"; do
      set -- $prm
      J=$1
      rho=$2
      echo "Launching: $model J=$J rho=$rho"
      python LLM_buyer.py --model "$model" --J $J --rho $rho --ratio $r --Nt $Nt > logs/RQ3_${model//\//_}_r${r}_J${J}_rho${rho}.log 2>&1 &
    done
  done
done

wait
echo "All experiments finished."
