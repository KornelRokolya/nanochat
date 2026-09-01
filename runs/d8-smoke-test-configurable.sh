#!/usr/bin/env bash

# Stop the script immediately if any command fails.
set -e

# Training configuration.
DEPTH=8
DEVICE_BATCH_SIZE=16
TOTAL_BATCH_SIZE=524288
NUM_ITERATIONS=100
WINDOW_PATTERN="L"

# Dataset configuration.
TOKENIZER_SHARDS=8
TRAINING_SHARDS=170


# =============================================================================
# CONFIGURATION
# Edit experiment-specific settings here.
# =============================================================================

# Git repository to clone.
# Replace this with your own fork.
REPO_URL="https://github.com/KornelRokolya/nanochat.git"

# Branch to use.
BRANCH="cyclic-lr"

# Main RunPod workspace.
WORKSPACE="/workspace"

# Local nanochat checkout.
REPO_DIR="${WORKSPACE}/nanochat"

# Nanochat datasets/tokenizers/checkpoints/cache.
NANOCHAT_BASE_DIR="${WORKSPACE}/nanochat_cache"

# Human-readable experiment identifier.
# This is also used to build the log filename.
EXPERIMENT_NAME="d8-baseline-${NUM_ITERATIONS}iter-3090"

# Nanochat run name.
# "dummy" disables online Weights & Biases logging.
RUN_NAME="dummy"

# Output log file.
OUTPUT_FILE="${WORKSPACE}/${EXPERIMENT_NAME}.txt"

# Find the first unused numbered log filename:
# d8-baseline_001.txt, d8-baseline_002.txt, ...
RUN_INDEX=1

while true; do
  RUN_NUMBER=$(printf "%03d" "$RUN_INDEX")
  OUTPUT_FILE="${WORKSPACE}/${EXPERIMENT_NAME}_${RUN_NUMBER}.txt"

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
echo "Repository:       ${REPO_URL}"
echo "Branch:           ${BRANCH}"
echo "Experiment:       ${EXPERIMENT_NAME}"
echo "Run name:         ${RUN_NAME}"
echo "Output file:      ${OUTPUT_FILE}"
echo "Depth:            ${DEPTH}"
echo "Iterations:       ${NUM_ITERATIONS}"
echo "Device batch:     ${DEVICE_BATCH_SIZE}"
echo "Total batch:      ${TOTAL_BATCH_SIZE}"
echo "Window pattern:   ${WINDOW_PATTERN}"
echo


# =============================================================================
# SYSTEM SETUP
# =============================================================================

cd "${WORKSPACE}"

# Refresh Ubuntu's package list.
apt-get update

# Install tools used by this script.
# tmux lets the training session keep running if JupyterLab/browser disconnects.
apt-get install -y curl git tmux


# Clone nanochat if it is not already present.
if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi


# Enter the repository.
cd "${REPO_DIR}"


# Fetch new commits/branches and switch to the configured branch.
git fetch origin
git switch "${BRANCH}"

# Bring the local branch up to date without creating an accidental merge commit.
git pull --ff-only


# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================

# Install uv if it is not already available.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Make uv visible in this shell.
export PATH="$HOME/.local/bin:$PATH"


# Create nanochat's virtual environment.
uv venv

# Install/synchronize all GPU dependencies.
uv sync --extra gpu

# Activate nanochat's virtual Python environment.
source .venv/bin/activate


# Tell nanochat where to put large persistent files.
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR}"
mkdir -p "${NANOCHAT_BASE_DIR}"


# =============================================================================
# GPU CHECK
# =============================================================================

echo "=== GPU / PyTorch check ==="

nvidia-smi

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 1024**3)"


# =============================================================================
# DATA + TOKENIZER
# =============================================================================

echo "=== Download first ${TOKENIZER_SHARDS} shards ==="
python -m nanochat.dataset -n "${TOKENIZER_SHARDS}"


# Tokenizer training - skipped if tokenizer found
TOKENIZER_DIR="${NANOCHAT_BASE_DIR}/tokenizer"

if [ -d "${TOKENIZER_DIR}" ]; then
  echo "=== Existing tokenizer found, skipping tokenizer training ==="
else
  echo "=== Download first ${TOKENIZER_SHARDS} shards ==="
  python -m nanochat.dataset -n "${TOKENIZER_SHARDS}"

  echo "=== Train tokenizer ==="
  python -m scripts.tok_train
fi


echo "=== Evaluate tokenizer ==="
python -m scripts.tok_eval


echo "=== Download full ${TRAINING_SHARDS}-shard dataset ==="
python -m nanochat.dataset -n "${TRAINING_SHARDS}"


# =============================================================================
# TRAINING
# =============================================================================

echo "=== Run ${EXPERIMENT_NAME} ==="

# 2>&1 combines normal output and errors.
# tee shows the output live while also saving it to OUTPUT_FILE.
python -m scripts.base_train2 \
  --depth="${DEPTH}" \
  --device-batch-size="${DEVICE_BATCH_SIZE}" \
  --total-batch-size="${TOTAL_BATCH_SIZE}" \
  --num-iterations="${NUM_ITERATIONS}" \
  --window-pattern="${WINDOW_PATTERN}" \
  --run="${RUN_NAME}" \
  --lr-cycle-amplitude=0.00 \
  --lr-cycle-period=40


echo
echo "=== Finished ==="
echo "Log saved to: ${OUTPUT_FILE}"
