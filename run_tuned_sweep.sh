#!/bin/sh
#SBATCH --job-name tuned-sweep
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mem=32000
#SBATCH --partition shared-gpu,public-gpu

#SBATCH --time 12:00:00

# Sweep of the 'tuned' sampler (prior 'dirichlet') in a single run: every combination of NFE (number of sampling
# steps T), ETA and LAMBD (see sample_tuned_sweep.py). Submit from the repo root: `mkdir -p sbatch && sbatch run_tuned_sweep.sh`.
# The results go to OUTPUT after each configuration; resubmitting the same job resumes the sweep.

DATASET="zinc"
DENOISER_DIR="./checkpoints/zinc/dirichlet6_state-a73worv0/files/"
DIRICHLET_SCALE="6"            # must match the checkpoint
SCHEDULE="cosine"              # cosine, linear or sparse, must match the checkpoint
INPUT="state"             # must match the checkpoint
NFE="128"                # numbers of sampling steps T, e.g. "32 64 128" (empty: model.T of the config)
ETA="0"           # 0.05 0.25 0.5 0.75 1
LAMBD="0 0.1 1 10 100 -1"      # -1: lambda = infinity (average weighted by the prediction)
N_SAMPLES="10000"              # graphs per configuration
OUTPUT="./sweeps/tuned_${DATASET}_${SCHEDULE}_scale${DIRICHLET_SCALE}.json"

srun apptainer exec --nv graph2.sif python3 ./sample_tuned_sweep.py \
    --dataset "${DATASET}" \
    --denoiser_dir "${DENOISER_DIR}" \
    --dirichlet_scale "${DIRICHLET_SCALE}" \
    --schedule "${SCHEDULE}" \
    --input "${INPUT}" \
    ${NFE:+--nfe ${NFE}} \
    --eta ${ETA} \
    --lambd ${LAMBD} \
    --n_samples "${N_SAMPLES}" \
    --output "${OUTPUT}" \
    --wandb on
