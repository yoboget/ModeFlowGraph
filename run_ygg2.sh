#!/bin/sh
#SBATCH --job-name jobname
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_titan_rtx:1
#SBATCH --cpus-per-task 1
#SBATCH --mem=32000
#SBATCH --partition shared-gpu,public-gpu


#SBATCH --time 12:00:00

srun apptainer exec --nv graph2.sif python3 ./main.py
