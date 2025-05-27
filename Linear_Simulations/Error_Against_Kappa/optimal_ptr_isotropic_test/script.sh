#!/bin/bash
# optimal_ptr_d120_spike_5avgtest.sbatch
# 
#SBATCH --job-name=optimal_ptr_d120_spike_5avgtest
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/Linear_Simulations/Error_Against_Kappa/optimal_ptr_isotropic_test/outputs/optimal_ptr_d120_spike_5avgtest_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/Linear_Simulations/Error_Against_Kappa/optimal_ptr_isotropic_test/outputs/optimal_ptr_d120_spike_5avgtest_%a.err
#SBATCH --array=1-21
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="runs"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python run.py $newdir 120 0.4 5 5 $SLURM_ARRAY_TASK_ID 0