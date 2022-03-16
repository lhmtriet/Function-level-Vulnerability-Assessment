#!/bin/bash
#SBATCH -p batch
#SBATCH --nodes 1
#SBATCH -c 16
#SBATCH --time=20:00:00
#SBATCH --mem=8GB
#SBATCH --array=1-315
#SBATCH --err="_logs/ml_results_single_%a.err"
#SBATCH --output="_logs/ml_results_single_%a.out"
#SBATCH --job-name="MyJobArray"

## Setup Python Environment
module load arch/haswell
module load Anaconda3/2020.07
module load CUDA/10.2.89
module load Java
module load Singularity
module load git/2.21.0-foss-2016b

source activate CVEfixes
source deactivate CVEfixes
source activate CVEfixes

## Echo job id into results file
echo "array_job_index: $SLURM_ARRAY_TASK_ID"

## Read inputs
IFS=',' read -ra par <<< `sed -n ${SLURM_ARRAY_TASK_ID}p Code/single/evaluate_models_single.csv`
python3 -u Code/single/evaluate_models_single.py "${par[0]}" "${par[1]}" "${par[2]}" "${par[3]}"