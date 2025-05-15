#!/bin/bash
# try_2layer_d20_fullbatch.sbatch
# 
#SBATCH --job-name=try_2layer_d20_fullbatch
#SBATCH -c 20
#SBATCH -t 23:00:00
#SBATCH -p kempner
#SBATCH --gres=gpu:1
#SBATCH --mem=300GB
#SBATCH -o /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/NONLINEAR/experiment/remote/kappasweep_neurips/outputs/try_2layer_d20_fullbatch_%a.out
#SBATCH -e /n/netscratch/pehlevan_lab/Lab/ml/ICL-structured-data/NONLINEAR/experiment/remote/kappasweep_neurips/outputs/try_2layer_d20_fullbatch_%a.err
#SBATCH --array=1-20%12
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

calculate_indices() {
    avgind=$(( ($1 - 1) / 5 ))
    kappaind=$(( ($1 - 1) % 5 ))
}
calculate_indices $SLURM_ARRAY_TASK_ID

parentdir="runs"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python kappasweep.py 20 $newdir $kappaind $avgind