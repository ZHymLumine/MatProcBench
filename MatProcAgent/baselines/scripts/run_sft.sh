#!/usr/bin/env bash
# ============================================================================
# run_sft.sh – Evidence-Augmented SFT training + evaluation for MatProcBench
# ============================================================================
#
# Pipeline:
#   1. Fine-tune base model on train.jsonl with gold evidence context (LoRA)
#   2. Evaluate fine-tuned model:
#      (a) sft_zero_shot – fine-tuned model, no context (measures parametric gain)
#      (b) sft_few_shot  – fine-tuned model + task-stratified in-context examples
#      (c) sft_evidence  – fine-tuned model + gold evidence (upper bound)
#      (d) sft_rag       – fine-tuned model + RAG retrieval (realistic eval)
#
# Usage:
#   # Train on additional_split, then evaluate all modes
#   bash run_sft.sh --split additional_split
#
#   # Train only (skip eval)
#   bash run_sft.sh --split type_split --train_only
#
#   # Eval only with existing checkpoint, select specific modes
#   bash run_sft.sh --split additional_split --eval_only --checkpoint /path/to/ckpt
#   bash run_sft.sh --split additional_split --eval_only --checkpoint /path/to/ckpt \
#                   --eval_modes "zero_shot few_shot"
# ============================================================================

set -e

export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="HF_TOKEN"

# ── Parse args ────────────────────────────────────────────────────────────────
SPLIT="additional_split"
TRAIN_ONLY=0
EVAL_ONLY=0
CHECKPOINT=""
EVAL_MODES="zero_shot few_shot rag"  # default: run all eval modes

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)       SPLIT="$2";       shift 2 ;;
        --model)       BASE_MODEL="$2";  shift 2 ;;
        --checkpoint)  CHECKPOINT="$2";  shift 2 ;;
        --train_only)  TRAIN_ONLY=1;     shift ;;
        --eval_only)   EVAL_ONLY=1;      shift ;;
        --eval_modes)  EVAL_MODES="$2";  shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_DIR="${SCRIPT_DIR}/.."
EVAL_SCRIPT="${SCRIPT_DIR}/../../eval.py"
PROCESSED_ROOT="${SCRIPT_DIR}/../../../data/processed"
RESULT_ROOT="${BASELINE_DIR}/results"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SIMPLE_MODEL="${BASE_MODEL//\//-}"

TRAIN_FILE="${PROCESSED_ROOT}/${SPLIT}/train.jsonl"
DEV_FILE="${PROCESSED_ROOT}/${SPLIT}/dev.jsonl"
TEST_FILE="${PROCESSED_ROOT}/${SPLIT}/test.jsonl"

# LoRA checkpoints go to a dedicated ckpt/ folder (separate from results/).
CKPT_ROOT="${BASELINE_DIR}/ckpt_step/${SIMPLE_MODEL}/${SPLIT}"
SFT_LORA_DIR="${CKPT_ROOT}/sft_lora"

# Default eval checkpoint: best_checkpoint saved by dev-monitoring callback.
# Override with --checkpoint to use a specific path.
CKPT_DIR="${SFT_LORA_DIR}/best_checkpoint"
if [[ -n "${CHECKPOINT}" ]]; then
    CKPT_DIR="${CHECKPOINT}"
fi

# Result dirs
SFT_ZERO_SHOT_DIR="${RESULT_ROOT}/${SIMPLE_MODEL}/${SPLIT}/sft_zero_shot"
SFT_FEW_SHOT_DIR="${RESULT_ROOT}/${SIMPLE_MODEL}/${SPLIT}/sft_few_shot"
SFT_EVIDENCE_DIR="${RESULT_ROOT}/${SIMPLE_MODEL}/${SPLIT}/sft_evidence"
SFT_RAG_DIR="${RESULT_ROOT}/${SIMPLE_MODEL}/${SPLIT}/sft_rag"
EMB_CACHE="${SFT_RAG_DIR}/rag_emb_cache.pkl"

echo "======================================================================"
echo " MatProcBench SFT Training"
echo " Base model : ${BASE_MODEL}"
echo " Split      : ${SPLIT}"
echo " Train file : ${TRAIN_FILE}"
echo " Dev file   : ${DEV_FILE}"
echo " Test file  : ${TEST_FILE}"
echo " Checkpoint : ${CKPT_DIR}"
echo " Eval modes : ${EVAL_MODES}"
echo "======================================================================"

# ── Step 1: LoRA fine-tuning ──────────────────────────────────────────────────
if [[ "${EVAL_ONLY}" == "0" ]]; then
    echo ""
    echo ">>> [train] Fine-tuning with evidence-augmented SFT ..."
    python "${BASELINE_DIR}/train_sft.py" \
        --model          "${BASE_MODEL}" \
        --train_file     "${TRAIN_FILE}" \
        --output_dir     "${SFT_LORA_DIR}" \
        --max_steps      100000 \
        --eval_steps     1000\ \
        --warmup_steps   50 \
        --logging_steps  10 \
        --batch_size     8 \
        --grad_accum     4 \
        --lr             2e-4 \
        --lora_r         16 \
        --lora_alpha     32 \
        --max_seq_len    1024 \
        --upsample_tasks "B1_condition_prediction:2,D1_process_ordering:2" \
        --dev_file       "${DEV_FILE}" \
        --early_stopping_patience 5

    echo ">>> [train] Done. Best checkpoint saved to: ${SFT_LORA_DIR}/best_checkpoint"
fi

if [[ "${TRAIN_ONLY}" == "1" ]]; then
    echo ">>> [skip] Eval skipped (--train_only)."
    exit 0
fi

# ── Helper: run eval.py on a result file ──────────────────────────────────────
run_eval() {
    local RESULT_FILE="$1"
    local SAVE_JSON="${RESULT_FILE%.jsonl}_eval.json"
    python "${EVAL_SCRIPT}" \
        --result_file  "${RESULT_FILE}" \
        --model        "${CKPT_DIR}" \
        --mode         "mcq" \
        --per_task \
        --save_json    "${SAVE_JSON}"
}

# ── Step 2a: zero-shot with fine-tuned model ──────────────────────────────────
if [[ " ${EVAL_MODES} " == *" zero_shot "* ]]; then
    echo ""
    echo "======================================================================"
    echo " Eval (a): sft_zero_shot — fine-tuned model, no context"
    echo "======================================================================"
    mkdir -p "${SFT_ZERO_SHOT_DIR}"
    python "${BASELINE_DIR}/run_baselines.py" \
        --method     zero_shot \
        --model      "${CKPT_DIR}" \
        --data_file  "${TEST_FILE}" \
        --save_file  "${SFT_ZERO_SHOT_DIR}/results.jsonl"
    run_eval "${SFT_ZERO_SHOT_DIR}/results.jsonl"
fi

# ── Step 2b: few-shot with fine-tuned model ───────────────────────────────────
if [[ " ${EVAL_MODES} " == *" few_shot "* ]]; then
    echo ""
    echo "======================================================================"
    echo " Eval (b): sft_few_shot — fine-tuned model + task-stratified examples"
    echo "======================================================================"
    mkdir -p "${SFT_FEW_SHOT_DIR}"
    python "${BASELINE_DIR}/run_baselines.py" \
        --method     few_shot \
        --model      "${CKPT_DIR}" \
        --data_file  "${TEST_FILE}" \
        --train_file "${TRAIN_FILE}" \
        --save_file  "${SFT_FEW_SHOT_DIR}/results.jsonl" \
        --n_shots    3
    run_eval "${SFT_FEW_SHOT_DIR}/results.jsonl"
fi

# ── Step 2c: gold evidence (upper bound) ──────────────────────────────────────
if [[ " ${EVAL_MODES} " == *" evidence "* ]]; then
    echo ""
    echo "======================================================================"
    echo " Eval (c): sft_evidence — fine-tuned model + gold evidence (upper bound)"
    echo "======================================================================"
    mkdir -p "${SFT_EVIDENCE_DIR}"
    python "${BASELINE_DIR}/train_sft.py" \
        --eval_only \
        --model       "${CKPT_DIR}" \
        --train_file  "${TRAIN_FILE}" \
        --test_file   "${TEST_FILE}" \
        --output_dir  "${SFT_EVIDENCE_DIR}"
    cp "${SFT_EVIDENCE_DIR}/sft_eval_results.jsonl" "${SFT_EVIDENCE_DIR}/results.jsonl"
    run_eval "${SFT_EVIDENCE_DIR}/results.jsonl"
fi

# ── Step 2d: RAG retrieval (realistic setting) ────────────────────────────────
if [[ " ${EVAL_MODES} " == *" rag "* ]]; then
    echo ""
    echo "======================================================================"
    echo " Eval (d): sft_rag — fine-tuned model + RAG retrieval (realistic)"
    echo "======================================================================"
    mkdir -p "${SFT_RAG_DIR}"
    python "${BASELINE_DIR}/run_baselines.py" \
        --method     rag \
        --model      "${CKPT_DIR}" \
        --data_file  "${TEST_FILE}" \
        --train_file "${TRAIN_FILE}" \
        --save_file  "${SFT_RAG_DIR}/results.jsonl" \
        --rag_k      3 \
        --embedder   all-mpnet-base-v2 \
        --emb_cache  "${EMB_CACHE}"
    run_eval "${SFT_RAG_DIR}/results.jsonl"
fi


echo ""
echo "======================================================================"
echo " All done. Results under: ${RESULT_ROOT}/${SIMPLE_MODEL}/${SPLIT}/"
echo "  sft_zero_shot : sft_zero_shot/results_eval.json"
echo "  sft_few_shot  : sft_few_shot/results_eval.json"
echo "  sft_evidence  : sft_evidence/results_eval.json"
echo "  sft_rag       : sft_rag/results_eval.json"
echo "======================================================================"
