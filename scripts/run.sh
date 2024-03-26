#!/bin/bash
#SBATCH --job-name=msc_pr_train
#SBATCH --output=logs/run_info/train_%j.log
#SBATCH --qos=normal
#SBATCH --time=8:00:00
#SBATCH --partition=cpu
#SBATCH --mem=16G
export PYTHONPATH=${PWD}

python -u train_agents.py
