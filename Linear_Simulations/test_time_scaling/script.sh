#!/bin/bash
# testime_d120_option2_5average.sbatch
# 
#SBATCH --job-name=testime_d120_option2_5average
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/Linear_Simulations/Error_Against_Kappa/test_time_scaling/outputs/testime_d120_option2_5average_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/Linear_Simulations/Error_Against_Kappa/test_time_scaling/outputs/testime_d120_option2_5average_%a.err
#SBATCH --array=1
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="runs"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python run.py $newdir 120 2 4 5 $SLURM_ARRAY_TASK_ID 2