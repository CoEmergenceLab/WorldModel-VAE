#!/bin/bash
#SBATCH -J VAE_model 
#SBATCH -o VAE_model_%j.o
#SBATCH -e VAE_model_%j.e
#SBATCH --mail-user=bp4297@rit.edu
#SBATCH --mail-type=ALL
#SBATCH -t 0-0:5:0
#SBATCH -A gwbc21 -p tier3
##BATCH -n 9
#SBATCH -N 1
#SBATCH --gres=gpu:p4:1
#SBATCH --mem=300g
spack load /vci5375
spack load /tjie5x3
spack load opencv

python3 WorldModel_VAE-SPORC.py