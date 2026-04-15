#!/bin/bash
#SBATCH --job-name=uniqat_batch
#SBATCH --output=logs/assess_batch_%j.log
#SBATCH --error=logs/assess_batch_%j.log
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# Activate your environment
# Handle Slurm spooling: use SLURM_SUBMIT_DIR if available
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    # Check if submitted from scripts directory
    if [[ "$SLURM_SUBMIT_DIR" == */scripts ]]; then
        PROJECT_ROOT="$(dirname "$SLURM_SUBMIT_DIR")"
    else
        PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    fi
else
    # Local execution
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

source "$PROJECT_ROOT/.venv/bin/activate"

# Set your input and output paths
INPUT_DIR="/path/to/your/survey/images"

# Create a unique output directory name based on the input path
INPUT_NAME="$(basename "$(dirname "$INPUT_DIR")")_$(basename "$INPUT_DIR")"
OUTPUT_DIR="/path/to/your/results/${INPUT_NAME}"

# Run assessment
python "$PROJECT_ROOT/scripts/assess_batch.py" \
    --paths "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --csv "$OUTPUT_DIR/assessment_results.csv" \
    --save-labels "$OUTPUT_DIR/training_labels.json" \
    --scale 0.2

exit_code=$?
echo "---- Finished with exit code $exit_code ----"
