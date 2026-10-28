#!/usr/bin/env bash
#SBATCH --job-name=tinyunet-train
#SBATCH --output=reports/slurm-%j.out
#SBATCH --error=reports/slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# ==============================================================================
# Environment variable overrides:
#   EPOCHS=50 BATCH_SIZE=16 LR=1e-3 CUDA_WHEEL=cu124 sbatch train.sh
# ==============================================================================

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-3}"
BASE_CHANNELS="${BASE_CHANNELS:-8}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.2}"
SNR_LOW="${SNR_LOW:--5.0}"
SNR_HIGH="${SNR_HIGH:-15.0}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/tinyunet.pth}"
CUDA_WHEEL="${CUDA_WHEEL:-cu124}"

mkdir -p reports checkpoints

echo "============================================================"
echo " Job ID          : ${SLURM_JOB_ID:-local}"
echo " Node            : $(hostname)"
echo " Date            : $(date --iso-8601=seconds)"
echo " Working Dir     : $(pwd)"
echo " Epochs          : ${EPOCHS}"
echo " Batch Size      : ${BATCH_SIZE}"
echo " Learning Rate   : ${LR}"
echo " DataLoader Wrks : ${NUM_WORKERS}"
echo " Checkpoint Path : ${CHECKPOINT_PATH}"
echo "============================================================"

# 1. Inspect NVIDIA GPU hardware
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[1/5] Verificando GPU NVIDIA via nvidia-smi..."
    nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
else
    echo "[!] AVISO: nvidia-smi não encontrado no PATH. Verifique se o job alocou GPU (--gres=gpu:1)."
fi

# 2. Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
    export PATH="${HOME}/.local/bin:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "[!] ERRO: 'uv' não encontrado. Instale com: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# 3. Sync base environment and install CUDA-enabled PyTorch if needed
echo "[2/5] Sincronizando ambiente Python via uv..."
uv sync

if ! uv run --no-sync python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[i] PyTorch atual não possui suporte CUDA ativo. Instalando PyTorch com CUDA (${CUDA_WHEEL})..."
    uv pip install --reinstall torch --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"
fi

# Verify CUDA is active inside PyTorch
uv run --no-sync python - <<'EOF'
import sys, torch
print(f" - PyTorch version : {torch.__version__}")
print(f" - CUDA available  : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("[!] ERRO: torch.cuda.is_available() retornou False mesmo após instalar a wheel CUDA.", file=sys.stderr)
    sys.exit(1)
print(f" - CUDA version    : {torch.version.cuda}")
print(f" - Device count    : {torch.cuda.device_count()}")
print(f" - GPU 0           : {torch.cuda.get_device_name(0)}")
EOF

# 4. Precompute dataset if metadata.csv or .pt files are missing
echo "[3/5] Verificando dataset pré-computado..."
if [ ! -f "data/precomputed/metadata.csv" ] || [ -z "$(find data/precomputed -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null)" ]; then
    echo "[i] Dataset pré-computado não encontrado. Executando preprocess..."
    uv run --no-sync noise_suppressor preprocess \
        --snr_low "${SNR_LOW}" \
        --snr_high "${SNR_HIGH}" \
        --seed "${SEED}"
else
    echo "[i] Dataset pré-computado já presente em data/precomputed/."
fi

# 5. Run GPU training
# Note: --no-sync prevents uv from reverting the CUDA torch wheel back to the CPU lockfile entry
echo "[4/5] Iniciando treinamento da TinyU-Net em GPU (CUDA)..."
uv run --no-sync noise_suppressor train \
    --device cuda \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --base_channels "${BASE_CHANNELS}" \
    --validation_ratio "${VALIDATION_RATIO}" \
    --snr_low "${SNR_LOW}" \
    --snr_high "${SNR_HIGH}" \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --checkpoint_path "${CHECKPOINT_PATH}"

# 6. Evaluate trained weights on validation split
echo "[5/5] Avaliando pesos treinados no split de validação..."
uv run --no-sync noise_suppressor evaluate \
    --device cuda \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --max_samples 100 \
    --seed "${SEED}"

echo "============================================================"
echo " [+] Treinamento concluído com sucesso!"
echo " [+] Pesos salvos em: ${CHECKPOINT_PATH}"
echo " Para usar os pesos localmente na sua máquina (CPU ou GPU):"
echo "   scp <usuario>@<cluster>:$(pwd)/${CHECKPOINT_PATH} ./checkpoints/tinyunet.pth"
echo "   uv run noise_suppressor demo"
echo "   uv run noise_suppressor infer --input <arquivo_ruidoso.wav>"
echo "============================================================"
