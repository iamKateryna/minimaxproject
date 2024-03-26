#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/run_info/eval_%j.log
#SBATCH --qos=normal
#SBATCH --time=5:00:00
#SBATCH --partition=cpu
#SBATCH --mem=8G
export PYTHONPATH=${PWD}

python -u evaluate_agents.py
