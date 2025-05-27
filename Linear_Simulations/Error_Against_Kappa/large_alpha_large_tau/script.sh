#!/bin/bash
# NIPS_piano_large80_d80.sbatch
# 
#SBATCH --job-name=NIPS_piano_large80_d80
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/Linear_Simulations/Error_Against_Kappa/large_alpha_large_tau/outputs/NIPS_piano_large80_d80_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/Linear_Simulations/Error_Against_Kappa/large_alpha_large_tau/outputs/NIPS_piano_large80_d80_%a.err
#SBATCH --array=1-21
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="runs"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python run.py $newdir 80 80 80 30 $SLURM_ARRAY_TASK_ID