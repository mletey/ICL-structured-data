#!/bin/bash
# NIPS_piano_spike_d130.sbatch
# 
#SBATCH --job-name=NIPS_piano_spike_d130
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/linear-ICL-structured-data/Linear_Simulations/Error_Against_Kappa/powerlaw/outputs/NIPS_piano_spike_d130_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/linear-ICL-structured-data/Linear_Simulations/Error_Against_Kappa/powerlaw/outputs/NIPS_piano_spike_d130_%a.err
#SBATCH --array=1-21
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="runs"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python run.py $newdir 130 2 4 10 $SLURM_ARRAY_TASK_ID