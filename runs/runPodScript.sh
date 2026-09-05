#!/usr/bin/env bash

# Stop immediately if any command fails, an unset variable is used,
# or a command inside a pipeline fails.
set -euo pipefail


# =============================================================================
# EXPERIMENT CONFIGURATION
# Edit this section for each run.
# =============================================================================

# Which trainer to run:
#   "original" -> scripts/base_train.py       (Karpathy baseline)
#   "new"      -> scripts/base_train_new.py   (new experimental trainer)
TRAINER="new"

# Give every experiment an explicit, stable name.
# Examples:
#   RTX3090_d4_baseline
#   RTX3090_d4_newCodeTest
#   RTX3090_d4_batchSizeIncreaseTest
#   RTX3090_d4d8_morphingTest
EXPERIMENT_NAME="RTX3090_d4_newCodeTest"

# Basic training configuration shared by BOTH trainers.
DEPTH=4
DEVICE_BATCH_SIZE=16

# -1 lets Nanochat calculate the model-size-dependent batch automatically.
# For controlled smoke tests you can instead specify a token batch explicitly,
# e.g. 32768 for d4,b16 with one accumulation step at sequence length 2048.
TOTAL_BATCH_SIZE=-1

# Keep this deliberately large when NEW trainer wall-clock stopping is enabled.
# The original trainer has no wall-clock stop, so for original runs this value
# is the actual training horizon and must be chosen appropriately.
NUM_ITERATIONS=1000000

MAX_SEQ_LEN=2048
WINDOW_PATTERN="L"
GPU_TYPE="RTX_3090"

# "dummy" disables online Weights & Biases logging.
RUN_NAME="dummy"


# =============================================================================
# NEW TRAINER ONLY
# These values are ignored when TRAINER="original".
# =============================================================================

# Exact accumulated training-time budget in seconds.
MAX_WALL_CLOCK_TIME=300

# Disable Nanochat's LR warmdown/decay while retaining warmup.
DISABLE_LR_DECAY=false

# Model morphing.
# Negative MORPH_AT_SECONDS disables morphing.
MORPH_AT_SECONDS=-1
MORPH_TARGET_DEPTH=-1
MORPH_STRATEGY="duplicate"       # duplicate | new_capacity
MORPH_NOISE_GAIN="1e-3"
MORPH_SEED=12345

# Independent batch-size growth.
# Negative BATCH_GROWTH_INTERVAL_SECONDS disables it.
BATCH_GROWTH_INTERVAL_SECONDS=-1
BATCH_GROWTH_START_SECONDS=-1    # -1 => first growth after one interval
BATCH_GROWTH_FACTOR=2


# =============================================================================
# EVALUATION
# =============================================================================

# Use a very large interval so evaluation does not interrupt short timed runs.
# Both trainers still perform their normal final evaluation if their code is
# configured to do so.
EVAL_EVERY=1000000000
CORE_METRIC_EVERY=1000000000
SAMPLE_EVERY=-1
SAVE_EVERY=-1


# =============================================================================
# DATA / REPOSITORY CONFIGURATION
# =============================================================================

TOKENIZER_SHARDS=8
TRAINING_SHARDS=170

REPO_URL="https://github.com/KornelRokolya/nanochat.git"
BRANCH="batch-schedule"

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/nanochat"
NANOCHAT_BASE_DIR="${WORKSPACE}/nanochat_cache"


# =============================================================================
# SELECT TRAINER
# =============================================================================

case "${TRAINER}" in
  original)
    TRAIN_MODULE="scripts.base_train"
    TRAINER_LABEL="original"
    ;;
  new)
    TRAIN_MODULE="scripts.base_train_new"
    TRAINER_LABEL="new"
    ;;
  *)
    echo "ERROR: TRAINER must be either 'original' or 'new', got: ${TRAINER}" >&2
    exit 1
    ;;
esac


# =============================================================================
# OUTER RUN LOG
#
# The new trainer additionally creates:
#   ${NANOCHAT_BASE_DIR}/experiments/${EXPERIMENT_NAME}/
#       config.json
#       log.txt
#       metrics.csv
#       events.csv
#       final_results.json
#       checkpoints/
#       raw_params/
#
# This outer .txt captures setup output too (apt, git, dataset download, etc.).
# =============================================================================

RUN_INDEX=1

while true; do
  RUN_NUMBER=$(printf "%03d" "${RUN_INDEX}")
  OUTPUT_FILE="${WORKSPACE}/${EXPERIMENT_NAME}_${TRAINER_LABEL}_${RUN_NUMBER}.txt"

  if [ ! -e "${OUTPUT_FILE}" ]; then
    break
  fi

  RUN_INDEX=$((RUN_INDEX + 1))
done

exec > >(tee "${OUTPUT_FILE}") 2>&1


# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

echo "=== Experiment configuration ==="
echo "Repository:             ${REPO_URL}"
echo "Branch:                 ${BRANCH}"
echo "Trainer:                ${TRAINER} (${TRAIN_MODULE})"
echo "Experiment:             ${EXPERIMENT_NAME}"
echo "Run name:               ${RUN_NAME}"
echo "Outer log:              ${OUTPUT_FILE}"
echo "GPU label:              ${GPU_TYPE}"
echo "Depth:                  ${DEPTH}"
echo "Iterations:             ${NUM_ITERATIONS}"
echo "Max sequence length:    ${MAX_SEQ_LEN}"
echo "Device batch:           ${DEVICE_BATCH_SIZE}"
echo "Total batch:            ${TOTAL_BATCH_SIZE}"
echo "Window pattern:         ${WINDOW_PATTERN}"

if [ "${TRAINER}" = "new" ]; then
  echo "Max training time:      ${MAX_WALL_CLOCK_TIME} s"
  echo "Disable LR decay:       ${DISABLE_LR_DECAY}"
  echo "Morph at:               ${MORPH_AT_SECONDS} s"
  echo "Morph target depth:     ${MORPH_TARGET_DEPTH}"
  echo "Morph strategy:         ${MORPH_STRATEGY}"
  echo "Morph noise gain:       ${MORPH_NOISE_GAIN}"
  echo "Batch growth interval:  ${BATCH_GROWTH_INTERVAL_SECONDS} s"
  echo "Batch growth start:     ${BATCH_GROWTH_START_SECONDS} s"
  echo "Batch growth factor:    ${BATCH_GROWTH_FACTOR}x"
fi
echo


# =============================================================================
# SYSTEM SETUP
# =============================================================================

cd "${WORKSPACE}"

apt-get update
apt-get install -y curl git tmux

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

git fetch origin
git switch "${BRANCH}"
git pull --ff-only


# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

export PATH="$HOME/.local/bin:$PATH"

# Reuse the environment if it already exists; otherwise create it.
if [ ! -d ".venv" ]; then
  uv venv
fi

uv sync --extra gpu
source .venv/bin/activate

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR}"
mkdir -p "${NANOCHAT_BASE_DIR}"


# =============================================================================
# SANITY CHECK THE SELECTED TRAINER
# =============================================================================

TRAINER_FILE="${REPO_DIR}/${TRAIN_MODULE//./\/}.py"

if [ ! -f "${TRAINER_FILE}" ]; then
  echo "ERROR: selected trainer does not exist:"
  echo "  ${TRAINER_FILE}"
  exit 1
fi

echo "Selected trainer file: ${TRAINER_FILE}"


# =============================================================================
# GPU CHECK
# =============================================================================

echo "=== GPU / PyTorch check ==="

nvidia-smi

python -c "import torch; \
print('PyTorch:', torch.__version__); \
print('CUDA:', torch.version.cuda); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0)); \
print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 1024**3)"


# =============================================================================
# DATA + TOKENIZER
# =============================================================================

TOKENIZER_DIR="${NANOCHAT_BASE_DIR}/tokenizer"

if [ -d "${TOKENIZER_DIR}" ]; then
  echo "=== Existing tokenizer found, skipping tokenizer training ==="
else
  echo "=== Download first ${TOKENIZER_SHARDS} shards for tokenizer ==="
  python -m nanochat.dataset -n "${TOKENIZER_SHARDS}"

  echo "=== Train tokenizer ==="
  python -m scripts.tok_train

  echo "=== Evaluate tokenizer ==="
  python -m scripts.tok_eval
fi

echo "=== Download full ${TRAINING_SHARDS}-shard dataset ==="
python -m nanochat.dataset -n "${TRAINING_SHARDS}"


# =============================================================================
# BUILD TRAINING COMMAND
#
# Use a Bash array so optional flags can be added safely.
# =============================================================================

TRAIN_CMD=(
  python -m "${TRAIN_MODULE}"
  "--depth=${DEPTH}"
  "--device-batch-size=${DEVICE_BATCH_SIZE}"
  "--total-batch-size=${TOTAL_BATCH_SIZE}"
  "--max-seq-len=${MAX_SEQ_LEN}"
  "--window-pattern=${WINDOW_PATTERN}"
  "--run=${RUN_NAME}"
  "--num-iterations=${NUM_ITERATIONS}"
  "--eval-every=${EVAL_EVERY}"
  "--core-metric-every=${CORE_METRIC_EVERY}"
  "--sample-every=${SAMPLE_EVERY}"
  "--save-every=${SAVE_EVERY}"
)

# Only the new trainer understands these arguments.
if [ "${TRAINER}" = "new" ]; then
  TRAIN_CMD+=(
    "--experiment-name=${EXPERIMENT_NAME}"
    "--max-wall-clock-time=${MAX_WALL_CLOCK_TIME}"
    "--morph-at-seconds=${MORPH_AT_SECONDS}"
    "--morph-target-depth=${MORPH_TARGET_DEPTH}"
    "--morph-strategy=${MORPH_STRATEGY}"
    "--morph-noise-gain=${MORPH_NOISE_GAIN}"
    "--morph-seed=${MORPH_SEED}"
    "--batch-growth-interval-seconds=${BATCH_GROWTH_INTERVAL_SECONDS}"
    "--batch-growth-start-seconds=${BATCH_GROWTH_START_SECONDS}"
    "--batch-growth-factor=${BATCH_GROWTH_FACTOR}"
  )

  if [ "${DISABLE_LR_DECAY}" = true ]; then
    TRAIN_CMD+=("--disable-lr-decay")
  fi
fi


# =============================================================================
# TRAINING
# =============================================================================

echo
echo "=== Run ${EXPERIMENT_NAME} ==="
echo "Command:"
printf ' %q' "${TRAIN_CMD[@]}"
echo
echo

"${TRAIN_CMD[@]}"


# =============================================================================
# FINISH
# =============================================================================

echo
echo "=== Finished ==="
echo "Outer setup/training log: ${OUTPUT_FILE}"

if [ "${TRAINER}" = "new" ]; then
  EXPERIMENT_DIR="${NANOCHAT_BASE_DIR}/experiments/${EXPERIMENT_NAME}"
  echo "Experiment directory:    ${EXPERIMENT_DIR}"
  echo "Metrics CSV:             ${EXPERIMENT_DIR}/metrics.csv"
  echo "Event log:               ${EXPERIMENT_DIR}/events.csv"
  echo "Resolved config:         ${EXPERIMENT_DIR}/config.json"
  echo "Final results:           ${EXPERIMENT_DIR}/final_results.json"
else
  echo "Baseline checkpoint dir: ${NANOCHAT_BASE_DIR}/base_checkpoints/d${DEPTH}"
fi
