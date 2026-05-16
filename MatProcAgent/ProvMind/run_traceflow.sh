#!/usr/bin/env bash
# ============================================================================
# run_traceflow.sh – ProvMind inference + evaluation for MatProcBench
# ============================================================================
#
# Runs ProvMind on a single split: uses that split's train.jsonl as the
# retrieval index and evaluates on that split's test.jsonl.
#
# Usage:
#   bash run_traceflow.sh --split type_split
#   bash run_traceflow.sh --split "type_split additional_split year_split"
#   bash run_traceflow.sh --split additional_split --model Qwen/Qwen2.5-7B-Instruct
#   bash run_traceflow.sh --split year_split --infer_only
#   bash run_traceflow.sh --split type_split --eval_only
# ============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=4

# ── Parse args ────────────────────────────────────────────────────────────────
SPLITS="additional_split year_split type_split random_split"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="Qwen/Qwen2.5-7B-Instruct"

INFER_ONLY=0
EVAL_ONLY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)      SPLITS="$2"; shift 2 ;;
        --model)      MODEL="$2";  shift 2 ;;
        --infer_only) INFER_ONLY=1; shift ;;
        --eval_only)  EVAL_ONLY=1;  shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${SCRIPT_DIR}/../.."           # …/MatProcBench
DATA_ROOT="${ROOT}/data/processed"
RAW_FILE="${ROOT}/data/raw_data/MatPROV.jsonl"
EVAL_SCRIPT="${ROOT}/MatProcAgent/eval.py"

SIMPLE_MODEL="${MODEL//\//-}"

# Inference hyper-parameters
TOP_K=4
PLAN_TOKENS=96
ANSWER_TOKENS=48
# EXTRA_FLAGS="--load_in_4bit"   # uncomment for 4-bit quantisation
EXTRA_FLAGS=""

# ── Banner ────────────────────────────────────────────────────────────────────
echo "======================================================================"
echo " ProvMind — same-split evaluation"
echo " Splits     : ${SPLITS}"
echo " Model      : ${MODEL}"
echo "======================================================================"

# ── Helpers ───────────────────────────────────────────────────────────────────
run_inference() {
    local SPLIT="$1"
    local RESULTS_DIR="${SCRIPT_DIR}/results/${SIMPLE_MODEL}/${SPLIT}"
    local TRAIN_FILE="${DATA_ROOT}/${SPLIT}/train.jsonl"
    local TEST_FILE="${DATA_ROOT}/${SPLIT}/test.jsonl"
    local SAVE_FILE="${RESULTS_DIR}/${SPLIT}_traceflow.jsonl"

    echo ""
    echo "======================================================================"
    echo " [Inference] train=${SPLIT}  test=${SPLIT}"
    echo "======================================================================"

    if [[ ! -f "${TRAIN_FILE}" ]]; then
        echo ">>> SKIP — train file not found: ${TRAIN_FILE}"; return
    fi
    if [[ ! -f "${TEST_FILE}" ]]; then
        echo ">>> SKIP — test file not found: ${TEST_FILE}"; return
    fi

    mkdir -p "${RESULTS_DIR}"
    python -m MatProcAgent.ProvMind.run_traceflow \
        --train_file            "${TRAIN_FILE}" \
        --data_file             "${TEST_FILE}" \
        --raw_file              "${RAW_FILE}" \
        --save_file             "${SAVE_FILE}" \
        --model                 "${MODEL}" \
        --top_k                 "${TOP_K}" \
        --plan_max_new_tokens   "${PLAN_TOKENS}" \
        --answer_max_new_tokens "${ANSWER_TOKENS}" \
        ${EXTRA_FLAGS}

    echo ">>> [Inference] Done. Results saved to: ${SAVE_FILE}"
}

run_eval() {
    local SPLIT="$1"
    local RESULTS_DIR="${SCRIPT_DIR}/results/${SIMPLE_MODEL}/${SPLIT}"
    local SAVE_FILE="${RESULTS_DIR}/${SPLIT}_traceflow.jsonl"

    echo ""
    echo "======================================================================"
    echo " [Eval] ${SPLIT}"
    echo "======================================================================"

    if [[ ! -f "${SAVE_FILE}" ]]; then
        echo ">>> SKIP — result file not found: ${SAVE_FILE}"; return
    fi

    python "${EVAL_SCRIPT}" \
        --result_file "${SAVE_FILE}" \
        --model       "${MODEL}" \
        --mode        mcq \
        --per_task \
        --save_json   "${RESULTS_DIR}/${SPLIT}_traceflow_eval.json"
}

# ── Main ──────────────────────────────────────────────────────────────────────
if [[ "${EVAL_ONLY}" == "0" ]]; then
    for SPLIT in ${SPLITS}; do
        run_inference "${SPLIT}"
    done
fi

if [[ "${INFER_ONLY}" == "1" ]]; then
    echo ">>> [skip] Eval skipped (--infer_only)."
    exit 0
fi

for SPLIT in ${SPLITS}; do
    run_eval "${SPLIT}"
done

echo ""
echo "======================================================================"
echo " All done. Results under: ${SCRIPT_DIR}/results/${SIMPLE_MODEL}/"
echo "======================================================================"
