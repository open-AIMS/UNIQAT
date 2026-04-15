#!/bin/bash
#SBATCH --job-name=uniqat_multi
#SBATCH --output=logs/assess_batch_%A_%a.log
#SBATCH --error=logs/assess_batch_%A_%a.log
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-8

# ============================================================================
#  UNIQAT - Multi-Survey Batch Assessment
#  Processes multiple survey folders in parallel via SLURM array jobs.
#  Edit the NAMES, BRANCHES, ROOT, and OUTPUT_ROOT arrays for your data.
#  Submit with: sbatch scripts/assess_batch_paper.sh
# ============================================================================

# --- Project root detection ---
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    if [[ "$SLURM_SUBMIT_DIR" == */scripts ]]; then
        PROJECT_ROOT="$(dirname "$SLURM_SUBMIT_DIR")"
    else
        PROJECT_ROOT="$SLURM_SUBMIT_DIR"
    fi
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

source "$PROJECT_ROOT/.venv/bin/activate"

# --- Data root on HPC NFS (edit for your cluster) ---
ROOT="/path/to/your/survey/data"
OUTPUT_ROOT="/path/to/your/results"

# --- Survey folders (example: 9 datasets) ---
# Short names for output folders and CSV files
NAMES=(
    "survey_2025_jan09"
    "survey_2025_jan11"
    "survey_2025_jan13"
    "survey_2025_feb08"
    "survey_2025_feb10"
    "survey_2025_feb12"
    "survey_2024_jan18"
    "survey_2024_jan20"
    "survey_2025_dec15"
)

# Relative paths under ROOT for each survey
BRANCHES=(
    "trip_01/camera/2025-01-09/sequence_01/cam_1"
    "trip_01/camera/2025-01-11/sequence_02/cam_1"
    "trip_01/camera/2025-01-13/sequence_03/cam_1"
    "trip_02/camera/2025-02-08/sequence_01/cam_1"
    "trip_02/camera/2025-02-10/sequence_02/cam_1"
    "trip_02/camera/2025-02-12/sequence_03/cam_1"
    "trip_03/camera/20240118_sequence_01"
    "trip_03/camera/20240120_sequence_02"
    "trip_04/camera/20251215_sequence_01"
)

# --- Select current task from array index ---
IDX=${SLURM_ARRAY_TASK_ID:-0}
SURVEY_NAME="${NAMES[$IDX]}"
BRANCH="${BRANCHES[$IDX]}"
INPUT_DIR="${ROOT}/${BRANCH}"
OUTPUT_DIR="${OUTPUT_ROOT}/${SURVEY_NAME}"

echo "============================================"
echo "Survey:    ${SURVEY_NAME}"
echo "Input:     ${INPUT_DIR}"
echo "Output:    ${OUTPUT_DIR}"
echo "Array ID:  ${IDX}"
echo "============================================"

# --- Validate input directory ---
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# --- Create output directories ---
mkdir -p "$OUTPUT_DIR/assessment_results"
mkdir -p "$OUTPUT_DIR/training_labels"
mkdir -p "$OUTPUT_DIR/metadata"
mkdir -p logs

# --- Copy photo_log.csv if it exists (depth/GPS metadata) ---
PHOTO_LOG=""
if [ -f "$INPUT_DIR/photo_log.csv" ]; then
    PHOTO_LOG="$INPUT_DIR/photo_log.csv"
elif [ -f "$(dirname "$INPUT_DIR")/photo_log.csv" ]; then
    PHOTO_LOG="$(dirname "$INPUT_DIR")/photo_log.csv"
fi

if [ -n "$PHOTO_LOG" ]; then
    cp "$PHOTO_LOG" "$OUTPUT_DIR/metadata/photo_log.csv"
    echo "Copied photo_log.csv from: $PHOTO_LOG"
else
    echo "WARNING: No photo_log.csv found for ${SURVEY_NAME}"
fi

# --- Run assessment ---
python "$PROJECT_ROOT/scripts/assess_batch.py" \
    --paths "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --csv "$OUTPUT_DIR/assessment_results/${SURVEY_NAME}.csv" \
    --save-labels "$OUTPUT_DIR/training_labels/${SURVEY_NAME}.json" \
    --scale 0.2

exit_code=$?
echo "---- ${SURVEY_NAME} finished with exit code $exit_code ----"
