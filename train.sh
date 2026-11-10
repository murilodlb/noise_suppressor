#!/usr/bin/env bash
#SBATCH --job-name=tinyunet-train
#SBATCH --output=reports/slurm-%j.out
#SBATCH --error=reports/slurm-%j.err
#SBATCH --partition=gpu-8-v100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

# ==============================================================================
# GUIA DE EXPERIMENTOS NO HPC & PRESETS RECOMENDADOS:
# ==============================================================================
#
# Evolução após os experimentos de 08/Nov (20261108_* / slurm-2246*):
#  1. Expansão de Ruído Não-Estacionário (augment_noise): transforma os 10 arquivos
#     .webm base em +90 arquivos .wav sintéticos (100 arquivos no total) via
#     concatenação com crossfade, superposição multi-fonte, pitch/rate shift e LFO.
#  2. Nova arquitetura ResTinyUNet (res_tinyunet): adiciona codificação posicional
#     de frequência (CoordConv), blocos residuais com Squeeze-and-Excitation (SE),
#     bottleneck dilatado (campo receptivo 2x maior) e interpolação pré-sigmoid.
#  3. Novo ciclo Infer-Loss-Update (asym_mag_mse + AdamW + Warmup + SpecAugment):
#     combina ponderação por energia espectral, penalização assimétrica de Recall
#     para proteger a fala e termo L1 contra o platô de gradiente, mais perturbação
#     on-the-fly do espectrograma (--augment_spec) durante as 80 épocas.
#
# --- EXECUÇÃO PADRÃO (Novo Preset Recomendado: ResTinyUNet + asym_mag_mse + 100 ruídos) ---
#   sbatch train.sh
#
# --- PRESET 1: Comparação Direta de Arquiteturas com o Novo Ciclo (ch=16, SPC=3) ---
#   MODEL=tinyunet     BASE_CHANNELS=16 LOSS=asym_mag_mse sbatch train.sh
#   MODEL=ds_tinyunet  BASE_CHANNELS=16 LOSS=asym_mag_mse sbatch train.sh
#   MODEL=res_tinyunet BASE_CHANNELS=16 LOSS=asym_mag_mse sbatch train.sh
#   MODEL=res_tinyunet BASE_CHANNELS=24 LOSS=asym_mag_mse sbatch train.sh
#
# --- PRESET 2: Comparação de Eficiência TinyML Embarcado (ch=8) ---
#   MODEL=tinyunet     BASE_CHANNELS=8 LOSS=asym_mag_mse sbatch train.sh
#   MODEL=ds_tinyunet  BASE_CHANNELS=8 LOSS=asym_mag_mse sbatch train.sh
#   MODEL=res_tinyunet BASE_CHANNELS=8 LOSS=asym_mag_mse sbatch train.sh
#
# --- PRESET 3: Ablation Study das Funções de Perda na ResTinyUNet ---
# for loss_name in bce mse l1 bce_l1 mag_mse asym_mag_mse; do
#   sbatch --job-name="res-${loss_name}" \
#     --export=ALL,MODEL=res_tinyunet,LOSS="${loss_name}",BASE_CHANNELS=16,SCHEDULER=cosine,EPOCHS=80 \
#     train.sh
# done
#
# --- PRESET 4: Forçar Regeneração Completa do Banco de Ruídos + Dataset (.pt) ---
#   FORCE_AUGMENT_NOISE=1 FORCE_PREPROCESS=1 SAMPLES_PER_CLEAN=3 sbatch train.sh
# ==============================================================================

MODEL="${MODEL:-res_tinyunet}"
LOSS="${LOSS:-asym_mag_mse}"
SCHEDULER="${SCHEDULER:-cosine}"
PATIENCE="${PATIENCE:-6}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BASE_CHANNELS="${BASE_CHANNELS:-16}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.2}"
SNR_LOW="${SNR_LOW:--5.0}"
SNR_HIGH="${SNR_HIGH:-15.0}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
SAMPLES_PER_CLEAN="${SAMPLES_PER_CLEAN:-3}"
AUGMENT_NOISE="${AUGMENT_NOISE:-1}"
AUGMENT_NOISE_FILES="${AUGMENT_NOISE_FILES:-90}"
FORCE_AUGMENT_NOISE="${FORCE_AUGMENT_NOISE:-0}"
AUGMENT_SPEC="${AUGMENT_SPEC:-1}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
EXP_TAG="${MODEL}_ch${BASE_CHANNELS}_${LOSS}_${SCHEDULER}_ep${EPOCHS}_bs${BATCH_SIZE}_spc${SAMPLES_PER_CLEAN}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/${EXP_TAG}.pth}"
BENCHMARK_PATH="${BENCHMARK_PATH:-reports/benchmark/benchmark_${EXP_TAG}.json}"
CUDA_WHEEL="${CUDA_WHEEL:-cu124}"

mkdir -p reports checkpoints reports/benchmark

echo "============================================================"
echo " Job ID          : ${SLURM_JOB_ID:-local}"
echo " Node            : $(hostname)"
echo " Date            : $(date --iso-8601=seconds)"
echo " Working Dir     : $(pwd)"
echo " Experiment Tag  : ${EXP_TAG}"
echo " Model           : ${MODEL}"
echo " Base Channels   : ${BASE_CHANNELS}"
echo " Loss Function   : ${LOSS}"
echo " LR Scheduler    : ${SCHEDULER} (patience=${PATIENCE})"
echo " Epochs          : ${EPOCHS}"
echo " Batch Size      : ${BATCH_SIZE}"
echo " Learning Rate   : ${LR} (weight_decay=${WEIGHT_DECAY})"
echo " Samples / Clean : ${SAMPLES_PER_CLEAN}"
echo " Augment Noise   : ${AUGMENT_NOISE} (num_files=${AUGMENT_NOISE_FILES})"
echo " Augment Spec    : ${AUGMENT_SPEC}"
echo " DataLoader Wrks : ${NUM_WORKERS}"
echo " Checkpoint Path : ${CHECKPOINT_PATH}"
echo " Benchmark Path  : ${BENCHMARK_PATH}"
echo "============================================================"

# 1. Inspect NVIDIA GPU hardware
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[1/6] Verificando GPU NVIDIA via nvidia-smi..."
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

# 3. Sync base environment and install CUDA-enabled PyTorch if needed (race-condition safe)
if uv run --no-sync python -c "import torch, soxr, pesq, pystoi, imageio_ffmpeg, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    echo "[2/6] Ambiente Python + PyTorch CUDA já validado e pronto (pulando uv sync)."
else
    echo "[2/6] Sincronizando ambiente Python via uv..."
    uv sync
    echo "[i] Instalando PyTorch com suporte CUDA (${CUDA_WHEEL})..."
    uv pip install --reinstall-package torch torch --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"
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

# 4. Expand non-stationary noise bank (run once if missing or forced) and precompute dataset
echo "[3/6] Verificando banco de ruídos expandido e dataset pré-computado..."
NEED_PREPROCESS=0
if [ "${AUGMENT_NOISE}" = "1" ]; then
    if [ "${FORCE_AUGMENT_NOISE}" = "1" ] || [ ! -f "data/noise/aug_noise_0001.wav" ]; then
        echo "[i] Sintetizando ${AUGMENT_NOISE_FILES} novos arquivos de ruído não-estacionário em data/noise/..."
        uv run --no-sync noise_suppressor augment_noise \
            --noise_dir data/noise \
            --num_files "${AUGMENT_NOISE_FILES}" \
            --duration 10.0 \
            --seed "${SEED}"
        NEED_PREPROCESS=1
    else
        echo "[i] Banco de ruídos expandido (aug_noise_*.wav) já presente em data/noise/."
    fi
fi

if [ "${FORCE_PREPROCESS}" = "1" ]; then
    echo "[i] FORCE_PREPROCESS=1 definido. Regenerando dataset pré-computado..."
    NEED_PREPROCESS=1
elif [ ! -f "data/precomputed/metadata.csv" ] || [ -z "$(find data/precomputed -maxdepth 1 -name '*.pt' -print -quit 2>/dev/null)" ]; then
    echo "[i] Dataset pré-computado não encontrado. Executando preprocess..."
    NEED_PREPROCESS=1
elif grep -q "sample-000100.mp3" "data/precomputed/metadata.csv" 2>/dev/null; then
    echo "[i] Dataset pré-computado antigo detectado (contém arquivos silenciosos sem filtro). Regenerando..."
    NEED_PREPROCESS=1
elif [ "${AUGMENT_NOISE}" = "1" ] && ! grep -q "aug_noise_" "data/precomputed/metadata.csv" 2>/dev/null; then
    echo "[i] Dataset pré-computado atual ainda não inclui os novos ruídos aug_noise_*.wav. Regenerando..."
    NEED_PREPROCESS=1
fi

if [ "${NEED_PREPROCESS}" = "1" ]; then
    uv run --no-sync noise_suppressor preprocess \
        --snr_low "${SNR_LOW}" \
        --snr_high "${SNR_HIGH}" \
        --samples_per_clean "${SAMPLES_PER_CLEAN}" \
        --seed "${SEED}"
else
    echo "[i] Dataset pré-computado limpo e atualizado já presente em data/precomputed/."
fi

# 5. Run GPU training
# Note: --no-sync prevents uv from reverting the CUDA torch wheel back to the CPU lockfile entry
TRAIN_EXTRA_ARGS=()
if [ "${AUGMENT_SPEC}" = "1" ]; then
    TRAIN_EXTRA_ARGS+=(--augment_spec)
fi

echo "[4/6] Iniciando treinamento do modelo '${MODEL}' (ch=${BASE_CHANNELS}, loss='${LOSS}', scheduler='${SCHEDULER}') em GPU (CUDA)..."
uv run --no-sync noise_suppressor train \
    --model "${MODEL}" \
    --loss "${LOSS}" \
    --scheduler "${SCHEDULER}" \
    --scheduler_patience "${PATIENCE}" \
    --device cuda \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --base_channels "${BASE_CHANNELS}" \
    --validation_ratio "${VALIDATION_RATIO}" \
    --snr_low "${SNR_LOW}" \
    --snr_high "${SNR_HIGH}" \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    "${TRAIN_EXTRA_ARGS[@]}"

# 6. Evaluate trained weights on validation split
echo "[5/6] Avaliando pesos treinados no split de validação..."
uv run --no-sync noise_suppressor evaluate \
    --model "${MODEL}" \
    --device cuda \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --max_samples 100 \
    --seed "${SEED}"

# 7. Benchmark computational overhead on CPU
echo "[6/6] Executando benchmark de overhead computacional..."
uv run --no-sync noise_suppressor benchmark \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --device cpu \
    --repeats 50 \
    --output_path "${BENCHMARK_PATH}"

echo "============================================================"
echo " [+] Treinamento, avaliação e benchmark concluídos com sucesso!"
echo " [+] Pesos salvos em    : ${CHECKPOINT_PATH}"
echo " [+] Benchmark salvo em : ${BENCHMARK_PATH}"
echo " Para usar os pesos localmente na sua máquina (CPU ou GPU):"
echo "   scp <usuario>@<cluster>:$(pwd)/${CHECKPOINT_PATH} ./checkpoints/"
echo "   uv run noise_suppressor demo --checkpoint_path ${CHECKPOINT_PATH}"
echo "   uv run noise_suppressor infer --checkpoint_path ${CHECKPOINT_PATH} --input <arquivo_ruidoso.wav>"
echo "============================================================"
