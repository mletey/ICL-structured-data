#!/bin/bash
# purelinear_d25_1000steps.sbatch
#
#SBATCH --job-name=purelinear_d25_1000steps
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/NONLINEAR/experiment/remote/kappasweep_neurips/outputs/purelinear_d25_1000steps_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/NONLINEAR/experiment/remote/kappasweep_neurips/outputs/purelinear_d25_1000steps_%a.err
#SBATCH -c 4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:20:00
#SBATCH --array=1-40%12
#SBATCH -p kempner_h100
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module purge
module load python/3.10.12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

calculate_indices() {
    avgind=$(( ($1 - 1)/5 ))
    kappaind=$(( ($1 - 1) % 5 ))
}
calculate_indices $SLURM_ARRAY_TASK_ID

parentdir="runs"
newdir="$parentdir/${SLURM_JOB_NAME}"
pkldir="$parentdir/${SLURM_JOB_NAME}/pickles"
mkdir "$newdir"
mkdir "$pkldir"
python kappasweep.py 25 $newdir $kappaind $avgind