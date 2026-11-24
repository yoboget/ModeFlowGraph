#!/bin/sh
#SBATCH --job-name scale-sweep
#SBATCH --error sbatch/error.e%j
#SBATCH --output sbatch/out.o%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mem=32000
#SBATCH --partition shared-gpu,public-gpu

#SBATCH --time 12:00:00

# Sweep of dirichlet_scale for vector_field vs quantile_map in a single run: every combination of SCALES, NFE and
# SAMPLERS (see sample_scale_sweep.py), with the total sampling time (tables + generation) of each configuration.
# Submit from the repo root: `mkdir -p sbatch && sbatch run_scale_sweep.sh`. The results go to
# sweeps/<sampler>_<dataset>_<schedule>_scale<scale>.json after each configuration; resubmitting resumes the sweep.
# Plots: python misc/plots.py --plot scale scale_time

DATASET="qm9"
DENOISER_DIR="./checkpoints/qm9/dirichlet_sufficient_small-nu0v78a0/files/"   # one checkpoint for every scale: input sufficient, no time input
SCHEDULE="cosine"              # cosine or linear (vector_field does not support sparse)
INPUT="sufficient"             # must be sufficient (and the checkpoint trained without --time_input)
SCALES="1 2 3 4 5 6"           # dirichlet_scale values, e.g. "1 2 3 4 5 6"
NFE="128"                      # numbers of sampling steps T, e.g. "32 64 128"
SAMPLERS="vector_field quantile_map"
N_SAMPLES="10000"              # graphs per configuration
VF_PRECISION="float32"         # vector_field C factor: float32 (original DFM, underflows) or float64
PGD="false"                    # true: also compute the PolyGraph Discrepancy (~15 min per configuration on 10k molecules)

srun apptainer exec --nv graph2.sif python3 ./sample_scale_sweep.py \
    --dataset "${DATASET}" \
    --denoiser_dir "${DENOISER_DIR}" \
    --schedule "${SCHEDULE}" \
    --input "${INPUT}" \
    --scales ${SCALES} \
    ${NFE:+--nfe ${NFE}} \
    --samplers ${SAMPLERS} \
    --n_samples "${N_SAMPLES}" \
    --vf_precision "${VF_PRECISION}" \
    $( [ "${PGD}" = "true" ] && echo --pgd || echo --no-pgd ) \
    --wandb on
