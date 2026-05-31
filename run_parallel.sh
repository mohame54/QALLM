#!/bin/bash
# run_parallel.sh — drop-in replacement for running infer.py directly.
# Accepts every argument that infer.py accepts, plus --num_gpus (default: auto-detect).
# Internally splits the dataset across all GPUs, runs in parallel, then merges.
#
# Usage (same flags as infer.py, num_gpus is optional):
#   ./run_parallel.sh --adapter_path ./my_adapter --batch_sz 16
#   ./run_parallel.sh --adapter_path ./my_adapter --num_gpus 4 --batch_sz 16

set -e

# ── Auto-detect GPU count (can be overridden with --num_gpus) ────────────────
NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
SAVE_FILE="submission-Finetuned_Aya.csv"
ARGS=("$@")

# Parse --num_gpus and --save_file_path from forwarded args
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    case "${ARGS[$i]}" in
        --num_gpus)
            NUM_GPUS="${ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --save_file_path)
            SAVE_FILE="${ARGS[$((i+1))]}"; i=$((i+2)) ;;
        *)
            i=$((i+1)) ;;
    esac
done

if [ "$NUM_GPUS" -lt 2 ]; then
    echo "⚠️  Only 1 GPU detected — running single-process inference."
    python infer.py "$@"
    exit 0
fi

BASENAME="${SAVE_FILE%.csv}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Launching parallel inference across ${NUM_GPUS} GPUs"
echo "   Final output → ${SAVE_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PIDS=()
PART_FILES=()

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    PART_FILE="${BASENAME}_part${gpu_id}.csv"
    PART_FILES+=("$PART_FILE")
    echo "   GPU ${gpu_id} → ${PART_FILE}"

    CUDA_VISIBLE_DEVICES=$gpu_id python aya.py \
        "$@" \
        --save_file_path "$PART_FILE" \
        --gpu_id "$gpu_id" \
        --num_gpus "$NUM_GPUS" \
        2>&1 | sed "s/^/[GPU ${gpu_id}] /" &

    PIDS+=($!)
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Wait for all processes and collect exit codes ────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "❌ GPU ${i} process failed with exit code ${STATUS}."
        FAILED=1
    fi
done

if [ $FAILED -ne 0 ]; then
    echo "❌ One or more processes failed. Aborting merge."
    exit 1
fi

# ── Merge all part files into the final output ───────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔗 Merging ${NUM_GPUS} part files into ${SAVE_FILE} …"

PART_FILES_PY=$(printf '"%s",' "${PART_FILES[@]}")
PART_FILES_PY="[${PART_FILES_PY%,}]"

python - <<PYEOF
import pandas as pd

part_files = ${PART_FILES_PY}
parts = [pd.read_csv(f) for f in part_files]
merged = pd.concat(parts).reset_index(drop=True)
merged.to_csv("${SAVE_FILE}", index=False)
total = sum(len(p) for p in parts)
print(f"✅ Merged {' + '.join(str(len(p)) for p in parts)} = {total} rows → ${SAVE_FILE}")
PYEOF

# ── Clean up part files ───────────────────────────────────────────────────────
for PART_FILE in "${PART_FILES[@]}"; do
    rm -f "$PART_FILE"
done
echo "🗑️  Cleaned up part files."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Done!"