#!/bin/bash
#SBATCH --job-name=humor-simple
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:ampere_a40:2
#SBATCH --mem=128G
#SBATCH --partition=ai
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eozlu21@ku.edu.tr

# Usage examples:
#   sbatch run_simple_job.sbatch --split test --max_new_tokens 2048 --output_dir results_simple_2048
#   sbatch run_simple_job.sbatch --split test_hard --max_new_tokens 2048 --output_dir results_simple_2048
#   sbatch run_simple_job.sbatch --split test_very_hard --max_new_tokens 2048 --output_dir results_simple_2048
# Any arguments after the sbatch filename are forwarded to the Python CLI.

set -euo pipefail

# Create logs dir if missing
mkdir -p logs
mkdir -p results_simple_2048

# Activate micromamba env (adjust path if different on cluster)
if [ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]; then
  source "$HOME/micromamba/etc/profile.d/micromamba.sh"
fi
micromamba activate humor-eval-env || source activate humor-eval-env || echo "Env activate fallback proceed"

# Print environment info
echo "Python: $(which python)"
python -V

# Forward all arguments to run_simple
echo "Running: python -m humor_eval.run_simple $@"
python -m humor_eval.run_simple "$@"

# List output files
ls -lh results_simple_2048 || true
