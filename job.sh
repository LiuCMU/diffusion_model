#!/bin/bash
#SBATCH -p gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=8
#SBATCH -t 48:00:00
#SBATCH -J lin_diffusion
#SBATCH -A cwr109
#SBATCH --export=ALL

export SLURM_EXPORT_ENV=ALL

module purge
module load gpu
module load slurm

source /home/zhen1997/anaconda3/etc/profile.d/conda.sh
conda activate dalle2

cd "/home/zhen1997/diffusion_model"
wandb online
# python test_encoder_decoder.py
# python diffusion.py
# python latent_diffusion.py
# python con_latent_diffuion32.py
python con_latent_diffusion_lin.py