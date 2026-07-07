#!/bin/bash
#
# Run all ImmuKRONOS DINO + I-JEPA ablation experiments on PLGrid.
# Submits jobs in packs of 8 (PLGrid job limit), waits for completion
# before launching the next pack.
#
# Usage:
#   chmod +x run_ablations_plgrid.sh
#   ./run_ablations_plgrid.sh
#
# Requirements:
#   - PLGrid access with SLURM scheduler
#   - Conda environment "immuvis" with all dependencies
#   - Project data accessible from compute nodes

# ===== CONFIGURATION =====
PARTITION="${PARTITION:-plgrid-gpu-a100}"
ACCOUNT="${ACCOUNT:-plgimmuvis-gpu-a100}"
TIME="48:00:00"
GPUS=1
CPUS=8
MEM="64G"
CONDA_ENV="immuvis"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="${WORKDIR}/slurm_logs"
RESULTS_DIR="${WORKDIR}/results"

# ===== SECRET KEYS =====
# Load from secrets file (not tracked in git) or environment variable.
# Create .secrets file with: COMET_API_KEY=your_key_here
SECRETS_FILE="${WORKDIR}/.secrets"
if [ -f "$SECRETS_FILE" ]; then
    echo "Loading secrets from ${SECRETS_FILE}"
    set -a
    source "$SECRETS_FILE"
    set +a
fi

if [ -z "${COMET_API_KEY:-}" ]; then
    echo "WARNING: COMET_API_KEY not set. Logging to Comet.ml will be disabled."
    echo "  Set via: export COMET_API_KEY=your_key"
    echo "  Or create .secrets file: echo 'COMET_API_KEY=your_key' > .secrets"
fi

mkdir -p "$LOGDIR"
mkdir -p "$RESULTS_DIR"

PACK_SIZE=8

# ===== ALL EXPERIMENT CONFIGS =====
# Pack 1: DINO ViT + ConvNeXt (8 experiments)
# Pack 2: DINO Swin + ViM (8 experiments)
# Pack 3: I-JEPA all backbones (8 experiments)

DINO_CONFIGS=(
    # ViT
    "configs/immukronos_vit_v2.yaml"
    "configs/immukronos_vit_v2_virtues.yaml"
    "configs/immukronos_vit_v3.yaml"
    "configs/immukronos_vit_v3_virtues.yaml"
    # ConvNeXt
    "configs/immukronos_convnext_v2.yaml"
    "configs/immukronos_convnext_v2_virtues.yaml"
    "configs/immukronos_convnext_v3.yaml"
    "configs/immukronos_convnext_v3_virtues.yaml"
    # Swin
    "configs/immukronos_swin_v2.yaml"
    "configs/immukronos_swin_v2_virtues.yaml"
    "configs/immukronos_swin_v3.yaml"
    "configs/immukronos_swin_v3_virtues.yaml"
    # ViM (Mamba)
    "configs/immukronos_vim_v2.yaml"
    "configs/immukronos_vim_v2_virtues.yaml"
    "configs/immukronos_vim_v3.yaml"
    "configs/immukronos_vim_v3_virtues.yaml"
)

IJEPA_CONFIGS=(
    "configs/ijepa_vit.yaml"
    "configs/ijepa_vit_virtues.yaml"
    "configs/ijepa_convnext.yaml"
    "configs/ijepa_convnext_virtues.yaml"
    "configs/ijepa_swin.yaml"
    "configs/ijepa_swin_virtues.yaml"
    "configs/ijepa_vim.yaml"
    "configs/ijepa_vim_virtues.yaml"
)

# ===== HELPER FUNCTIONS =====

submit_job() {
    local CONFIG="$1"
    local SCRIPT="$2"
    local JOB_NAME
    JOB_NAME="$(basename "$CONFIG" .yaml)"

    local OUTPUT_LOG="${LOGDIR}/${JOB_NAME}_%j.out"
    local ERROR_LOG="${LOGDIR}/${JOB_NAME}_%j.err"

    local JOB_ID
    JOB_ID=$(sbatch --parsable \
        --job-name="$JOB_NAME" \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --time="$TIME" \
        --gres="gpu:${GPUS}" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --output="$OUTPUT_LOG" \
        --error="$ERROR_LOG" \
        --export=ALL \
        <<EOF
#!/bin/bash
#SBATCH --nice=0

echo "=========================================="
echo "Job: ${JOB_NAME}"
echo "Config: ${CONFIG}"
echo "Script: ${SCRIPT}"
echo "Node: \$(hostname)"
echo "GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: \$(date)"
echo "=========================================="

# Load environment
source ~/.bashrc
conda activate ${CONDA_ENV}
cd ${WORKDIR}

# Export secret keys
export COMET_API_KEY="${COMET_API_KEY:-}"

# Run training
python ${SCRIPT} ${CONFIG} 2>&1 | tee "${RESULTS_DIR}/${JOB_NAME}.log"

EXIT_CODE=\$?
echo "=========================================="
echo "End: \$(date)"
echo "Exit code: \${EXIT_CODE}"
echo "=========================================="

# Save summary
echo "${JOB_NAME},\${EXIT_CODE},\$(date +%Y%m%d_%H%M%S)" >> "${RESULTS_DIR}/completed_jobs.csv"
exit \${EXIT_CODE}
EOF
    )

    echo "  Submitted: ${JOB_NAME} (Job ID: ${JOB_ID})"
    echo "$JOB_ID"
}

wait_for_jobs() {
    local JOB_IDS=("$@")
    if [ ${#JOB_IDS[@]} -eq 0 ]; then
        return
    fi

    echo ""
    echo "Waiting for ${#JOB_IDS[@]} jobs to complete..."
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""

    # Build dependency string for squeue polling
    local ALL_DONE=false
    while [ "$ALL_DONE" = false ]; do
        ALL_DONE=true
        for JID in "${JOB_IDS[@]}"; do
            # Check if job is still in queue (PENDING/RUNNING/etc)
            local STATE
            STATE=$(squeue -j "$JID" -h -o "%T" 2>/dev/null)
            if [ -n "$STATE" ]; then
                ALL_DONE=false
                break
            fi
        done
        if [ "$ALL_DONE" = false ]; then
            sleep 60
        fi
    done

    echo "All jobs in pack completed."
    echo ""

    # Report results
    for JID in "${JOB_IDS[@]}"; do
        local JOB_STATE
        JOB_STATE=$(sacct -j "$JID" --format=JobName,State,ExitCode,Elapsed --noheader -P 2>/dev/null | head -1)
        echo "  $JOB_STATE"
    done
    echo ""
}

# ===== MAIN EXECUTION =====

echo "============================================"
echo " ImmuKRONOS Ablation Study — PLGrid Launcher"
echo "============================================"
echo "Total experiments: $((${#DINO_CONFIGS[@]} + ${#IJEPA_CONFIGS[@]}))"
echo "  DINO:  ${#DINO_CONFIGS[@]}"
echo "  I-JEPA: ${#IJEPA_CONFIGS[@]}"
echo "Pack size: ${PACK_SIZE}"
echo "Partition: ${PARTITION}"
echo "Account: ${ACCOUNT}"
echo "Work dir: ${WORKDIR}"
echo "============================================"
echo ""

# Initialize results CSV
echo "job_name,exit_code,timestamp" > "${RESULTS_DIR}/completed_jobs.csv"

PACK_NUM=0

# --- Submit DINO experiments in packs of 8 ---
echo "=== DINO Experiments (train_immukronos_unified.py) ==="
CURRENT_PACK=()
for CONFIG in "${DINO_CONFIGS[@]}"; do
    JOB_ID=$(submit_job "$CONFIG" "train_immukronos_unified.py")
    CURRENT_PACK+=("$JOB_ID")

    if [ ${#CURRENT_PACK[@]} -ge $PACK_SIZE ]; then
        PACK_NUM=$((PACK_NUM + 1))
        echo ""
        echo "--- Pack ${PACK_NUM} submitted (${#CURRENT_PACK[@]} jobs) ---"
        wait_for_jobs "${CURRENT_PACK[@]}"
        CURRENT_PACK=()
    fi
done

# Flush remaining DINO jobs
if [ ${#CURRENT_PACK[@]} -gt 0 ]; then
    PACK_NUM=$((PACK_NUM + 1))
    echo ""
    echo "--- Pack ${PACK_NUM} submitted (${#CURRENT_PACK[@]} jobs) ---"
    wait_for_jobs "${CURRENT_PACK[@]}"
    CURRENT_PACK=()
fi

# --- Submit I-JEPA experiments in packs of 8 ---
echo ""
echo "=== I-JEPA Experiments (train_ijepa.py) ==="
for CONFIG in "${IJEPA_CONFIGS[@]}"; do
    JOB_ID=$(submit_job "$CONFIG" "train_ijepa.py")
    CURRENT_PACK+=("$JOB_ID")

    if [ ${#CURRENT_PACK[@]} -ge $PACK_SIZE ]; then
        PACK_NUM=$((PACK_NUM + 1))
        echo ""
        echo "--- Pack ${PACK_NUM} submitted (${#CURRENT_PACK[@]} jobs) ---"
        wait_for_jobs "${CURRENT_PACK[@]}"
        CURRENT_PACK=()
    fi
done

# Flush remaining I-JEPA jobs
if [ ${#CURRENT_PACK[@]} -gt 0 ]; then
    PACK_NUM=$((PACK_NUM + 1))
    echo ""
    echo "--- Pack ${PACK_NUM} submitted (${#CURRENT_PACK[@]} jobs) ---"
    wait_for_jobs "${CURRENT_PACK[@]}"
fi

# ===== FINAL SUMMARY =====
echo ""
echo "============================================"
echo " ALL ABLATION EXPERIMENTS COMPLETED"
echo "============================================"
echo "Results saved to: ${RESULTS_DIR}/"
echo "  - Per-experiment logs: ${RESULTS_DIR}/<config_name>.log"
echo "  - Job summary: ${RESULTS_DIR}/completed_jobs.csv"
echo "  - SLURM logs: ${LOGDIR}/"
echo ""
echo "Completion summary:"
cat "${RESULTS_DIR}/completed_jobs.csv"
echo "============================================"
