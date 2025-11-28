#!/bin/bash
mkdir -p logs
Nt=400

params=(
  "0 0.8"
  "1.0 0.8"
  "1.0 0.1"
)
for prm in "${params[@]}"; do
  set -- $prm
  J=$1
  rho=$2
  echo "Launching: J=$J rho=$rho"
  python equation_based.py --J $J --rho $rho --Nt $Nt > logs/Equation_J${J}_rho${rho}.log 2>&1 &
done


wait
echo "All experiments finished."
