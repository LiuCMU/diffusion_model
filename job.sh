#!/bin/bash
#SBATCH -p gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=8
#SBATCH -t 2:00:00
#SBATCH -J decoder_encoder
#SBATCH -A cwr109
#SBATCH --export=ALL

export SLURM_EXPORT_ENV=ALL

module purge
module load gpu
module load slurm

source /home/zhen1997/anaconda3/etc/profile.d/conda.sh
conda activate dalle2

cd "/home/zhen1997/diffusion_model"
python test_encoder_decoder.py
