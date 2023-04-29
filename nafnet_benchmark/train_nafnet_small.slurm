#!/bin/bash
#
#SBATCH --time=04:00:00
#SBATCH --job-name=small_nafnet
#SBATCH --chdir=/home/katehu/diffusion/nafnet_benchmark
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=threedle-owned
#SBATCH --gres=gpu:1
#SBATCH --nodelist=b001
#SBATCH --cpus-per-task=4
#
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=katehu310@gmail.com

srun python -u train_nafnet_small.py
