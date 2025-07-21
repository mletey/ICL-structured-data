#!/bin/bash
# d100_lessfuckingnoiseagain.sbatch
# 
#SBATCH --job-name=d100_lessfuckingnoiseagain
#SBATCH -c 1
#SBATCH -t 1:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/Linear_Simulations/test_time_scaling/outputs/d100_lessfuckingnoiseagain_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/Linear_Simulations/test_time_scaling/outputs/d100_lessfuckingnoiseagain_%a.err
#SBATCH --array=1
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="runs"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python run.py $newdir 100 2 4 10 $SLURM_ARRAY_TASK_ID 0