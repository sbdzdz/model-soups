#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --partition=a100-galvani
#SBATCH --output=/mnt/lustre/work/bethge/dziadzio08/projects/specops/slurm/specops_%j.out
#SBATCH --error=/mnt/lustre/work/bethge/dziadzio08/projects/specops/slurm/specops_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.dziadzio@uni-tuebingen.de

scontrol show job $SLURM_JOB_ID

conda activate soups
source $HOME/.bashrc
export WORK='/mnt/lustre/work/bethge/dziadzio08'

additional_args="$@"
python $HOME/repos/soups/specops.py $additional_args