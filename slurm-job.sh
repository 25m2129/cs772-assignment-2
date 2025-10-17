#!/bin/bash
#SBATCH -J my_ml_job              # Job name
#SBATCH -o logs/%x_%j.out         # Standard output (%x=job name, %j=job ID)
#SBATCH -e logs/%x_%j.err         # Standard error
#SBATCH -N 1                      # Number of nodes
#SBATCH --ntasks-per-node=1       # Tasks per node
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=8         # Number of CPU cores
#SBATCH --mem=64G                 # Memory per node
#SBATCH --time=04:00:00           # Maximum runtime (hh:mm:ss)
#SBATCH --partition=a40           # GPU partition (use a100, a40, l4, or l40s)
#SBATCH --qos=a40                 # QoS (same as partition usually)

python ./main.py
