#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./submit.sh "python train.py --epochs 10"

CMD="${*:-}"
script="$2"

if [[ -z "$CMD" || -z "$script" ]]; then
  echo "Usage: $0 <command>"
  exit 1
fi

base="$(basename "$script")"
job_name="${base%.*}"

source .env

log_dir=logs
mkdir -p "$log_dir"
slurm_dir=./scripts/sb/tmp
mkdir -p "$slurm_dir"

TMP=$(mktemp "$slurm_dir"/slurm-job-XXXXXX.slurm)

cat > "$TMP" <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}_test
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=$log_dir/%x-%j.out
#SBATCH --error=$log_dir/%x-%j.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=$ACADEMIC_EMAIL 

# Activate our required virtual environment
source .venv/bin/activate

# Specify common cache for Transformers library
export HF_HOME="$HOME/.cache/huggingface"

srun $CMD
EOF

JOB_ID=$(sbatch "$TMP" | awk '{print $4}')
echo "Submitted job $JOB_ID"