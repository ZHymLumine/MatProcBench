#!/usr/bin/env bash
# ============================================================================
# run_cross_split.sh – Cross-split OOD evaluation for MatProcBench SFT models
# ============================================================================
#
# Trains (or loads) a model on a given split and evaluates it across all
# OOD-clean test splits to measure generalisation under three shift types:
#
#   additional_split → paper-level OOD   (unseen papers)
#   type_split       → material-type OOD (novel material chemistries)
#   year_split       → temporal OOD      (newer publications)
#
# Clean cross-split pairs (0% DOI contamination):
#   Train: additional → Test: additional, type, year   ← recommended
#   Train: type       → Test: additional, type
#   Train: year       → Test: additional, year
#
# Usage
# -----
#   # Train on additional_split, then evaluate all OOD splits (gold evidence)
#   bash run_cross_split.sh --train_split additional_split
#
#   # Use existing checkpoint (skip training)
#   bash run_cross_split.sh \
#       --train_split additional_split \
#       --checkpoint  /path/to/checkpoint \
#       --train_only 0
#
#   # Run all eval modes (zero_shot + gold_evidence + rag)
#   bash run_cross_split.sh \
#       --train_split additional_split \
#       --eval_mode   all
#
#   # Custom test splits (e.g. skip year_split)
#   bash run_cross_split.sh \
#       --train_split additional_split \
#       --test_splits "additional_split,type_split"
# ============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="${HF_TOKEN:-}"

# ── Defaults ──────────────────────────────────────────────────────────────────
TRAIN_SPLIT="additional_split"
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
CHECKPOINT="/home/yzhang/research/MatProcBench/MatProcAgent/baselines/ckpt_step/Qwen-Qwen2.5-7B-Instruct/additional_split/sft_lora/best_checkpoint"           # set to skip training
EVAL_MODE="zero_shot"
TRAIN_ONLY=0
EVAL_ONLY=0
MAX_STEPS=3000
BATCH_SIZE=4
GRAD_ACCUM=4
LORA_R=32
LORA_ALPHA=64
EVIDENCE_DROPOUT=0.0
BALANCE_STRATEGY="sqrt"
RAG_K=3

# Default test splits depend on train_split (set after arg parsing)
TEST_SPLITS="additional_split,type_split,year_split,random_split"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train_split)      TRAIN_SPLIT="$2";       shift 2 ;;
        --model)            BASE_MODEL="$2";         shift 2 ;;
        --checkpoint)       CHECKPOINT="$2";         shift 2 ;;
        --eval_mode)        EVAL_MODE="$2";          shift 2 ;;
        --test_splits)      TEST_SPLITS="$2";        shift 2 ;;
        --train_only)       TRAIN_ONLY=1;            shift ;;
        --eval_only)        EVAL_ONLY=1;             shift ;;
        --max_steps)        MAX_STEPS="$2";          shift 2 ;;
        --batch_size)       BATCH_SIZE="$2";         shift 2 ;;
        --grad_accum)       GRAD_ACCUM="$2";         shift 2 ;;
        --lora_r)           LORA_R="$2";             shift 2 ;;
        --lora_alpha)       LORA_ALPHA="$2";         shift 2 ;;
        --evidence_dropout) EVIDENCE_DROPOUT="$2";   shift 2 ;;
        --balance_strategy) BALANCE_STRATEGY="$2";   shift 2 ;;
        --rag_k)            RAG_K="$2";              shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Default OOD-clean test splits per training split ─────────────────────────
if [[ -z "${TEST_SPLITS}" ]]; then
    case "${TRAIN_SPLIT}" in
        additional_split) TEST_SPLITS="additional_split,type_split,year_split, random_split" ;;
        type_split)       TEST_SPLITS="additional_split,type_split" ;;
        year_split)       TEST_SPLITS="additional_split,year_split" ;;
        random_split)
            echo "WARNING: random_split.train contaminates all other test sets." >&2
            echo "         Cross-split OOD evaluation is NOT valid for random_split." >&2
            TEST_SPLITS="random_split"
            ;;
        *) echo "Unknown train_split: ${TRAIN_SPLIT}" >&2; exit 1 ;;
    esac
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_DIR="${SCRIPT_DIR}/.."
PROCESSED_ROOT="${SCRIPT_DIR}/../../../data/processed"
RESULT_ROOT="${BASELINE_DIR}/results"

SIMPLE_MODEL="${BASE_MODEL//\//-}"
CKPT_ROOT="${BASELINE_DIR}/ckpt_step/${SIMPLE_MODEL}/${TRAIN_SPLIT}"
SFT_LORA_DIR="${CKPT_ROOT}/sft_lora"

# Use provided checkpoint or fall back to best_checkpoint from training
if [[ -n "${CHECKPOINT}" ]]; then
    CKPT_DIR="${CHECKPOINT}"
else
    CKPT_DIR="${SFT_LORA_DIR}/best_checkpoint"
fi

CROSS_RESULT_DIR="${RESULT_ROOT}/cross_split/${SIMPLE_MODEL}/${TRAIN_SPLIT}"

TRAIN_FILE="${PROCESSED_ROOT}/${TRAIN_SPLIT}/train.jsonl"
DEV_FILE="${PROCESSED_ROOT}/${TRAIN_SPLIT}/dev.jsonl"

echo "======================================================================"
echo " MatProcBench Cross-Split Evaluation"
echo " Base model      : ${BASE_MODEL}"
echo " Train split     : ${TRAIN_SPLIT}"
echo " Test splits     : ${TEST_SPLITS}"
echo " Eval mode       : ${EVAL_MODE}"
echo " Checkpoint      : ${CKPT_DIR}"
echo " Results dir     : ${CROSS_RESULT_DIR}"
echo "======================================================================"

# ── Step 1: LoRA fine-tuning (skipped if --eval_only or --checkpoint given) ──
if [[ "${EVAL_ONLY}" == "0" && -z "${CHECKPOINT}" ]]; then
    if [[ -d "${CKPT_DIR}" ]]; then
        echo ""
        echo ">>> [train] Checkpoint found at ${CKPT_DIR}, skipping training."
        echo "    Delete it or pass --checkpoint to re-train."
    else
        echo ""
        echo ">>> [train] Fine-tuning on ${TRAIN_SPLIT} ..."
        python "${BASELINE_DIR}/train_sft.py" \
            --model               "${BASE_MODEL}" \
            --train_file          "${TRAIN_FILE}" \
            --dev_file            "${DEV_FILE}" \
            --output_dir          "${SFT_LORA_DIR}" \
            --max_steps           "${MAX_STEPS}" \
            --eval_steps          200 \
            --warmup_steps        200 \
            --logging_steps       20 \
            --batch_size          "${BATCH_SIZE}" \
            --grad_accum          "${GRAD_ACCUM}" \
            --lr                  2e-4 \
            --lora_r              "${LORA_R}" \
            --lora_alpha          "${LORA_ALPHA}" \
            --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
            --max_seq_len         1024 \
            --balance_strategy    "${BALANCE_STRATEGY}" \
            --evidence_dropout    "${EVIDENCE_DROPOUT}" \
            --early_stopping_patience 5

        echo ">>> [train] Done. Best checkpoint: ${CKPT_DIR}"
    fi
fi

if [[ "${TRAIN_ONLY}" == "1" ]]; then
    echo ">>> [skip] Evaluation skipped (--train_only)."
    exit 0
fi

# ── Step 2: Cross-split evaluation ───────────────────────────────────────────
echo ""
echo ">>> [cross_eval] Evaluating checkpoint across test splits ..."
echo "    Checkpoint : ${CKPT_DIR}"
echo "    Test splits: ${TEST_SPLITS}"
echo "    Eval mode  : ${EVAL_MODE}"

mkdir -p "${CROSS_RESULT_DIR}"

python "${BASELINE_DIR}/eval_cross_split.py" \
    --checkpoint   "${CKPT_DIR}" \
    --train_split  "${TRAIN_SPLIT}" \
    --test_splits  "${TEST_SPLITS}" \
    --eval_mode    "${EVAL_MODE}" \
    --data_root    "${PROCESSED_ROOT}" \
    --output_dir   "${CROSS_RESULT_DIR}/${EVAL_MODE}" \
    --batch_size   8 \
    --rag_k        "${RAG_K}" \
    --embedder     "all-mpnet-base-v2"

echo ""
echo "======================================================================"
echo " Done. Results saved to: ${CROSS_RESULT_DIR}/${EVAL_MODE}/"
echo ""
echo " Key files:"
echo "   Summary JSON : ${CROSS_RESULT_DIR}/${EVAL_MODE}/cross_split_summary.json"
for mode_dir in "${CROSS_RESULT_DIR}/${EVAL_MODE}"/*/; do
    for split_dir in "${mode_dir}"*/; do
        [[ -f "${split_dir}results.jsonl" ]] && \
            echo "   ${split_dir}results.jsonl"
    done
done 2>/dev/null || true
echo "======================================================================"
