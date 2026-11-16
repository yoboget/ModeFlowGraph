#!/bin/sh
#SBATCH --job-name jobname
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task 1
#SBATCH --mem=32000
#SBATCH --partition private-kalousis-gpu,private-cui-gpu

#SBATCH --time 48:00:00

srun apptainer exec --nv graph.sif python3 ./main.py
