#!/bin/bash
# poster_tr1_test1_d200_FAST_test.sbatch
# 
#SBATCH --job-name=poster_tr1_test1_d200_FAST_test
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/linear-ICL-structured-data/Linear_Simulations/Error_Against_Kappa/powerlaw/outputs/poster_tr1_test1_d200_FAST_test_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/linear-ICL-structured-data/Linear_Simulations/Error_Against_Kappa/powerlaw/outputs/poster_tr1_test1_d200_FAST_test_%a.err
#SBATCH --array=1-10
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="runs"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python run.py $newdir 200 0.5 5 1 $SLURM_ARRAY_TASK_ID 1 1