#!/bin/bash
mkdir -p logs

# deepseek-v3, gpt-4o, gpt-5, gemini-2.5-flash-lite, Qwen/Qwen2.5-32B-Instruct
models=(
  "deepseek-v3"
)

Nt=400

params=(
  "0.5 0.8"
  "1.0 0.8"
  "1.0 0.1"
)
for model in "${models[@]}"; do
  for prm in "${params[@]}"; do
    set -- $prm
    J=$1
    rho=$2
    echo "Launching: $model J=$J rho=$rho"
    python LLM_seller.py --model "$model" --J $J --rho $rho --Nt $Nt > logs/LLM_seller_${model//\//_}_J${J}_rho${rho}.log 2>&1 &
  done
done

wait
echo "All experiments finished."
