#!/usr/bin/env bash
# run_cross_split_eval.sh
#
# Cross-split evaluation for TraceFlow:
#   - Train index: additional_split/train.jsonl
#   - Evaluate on: additional_split / year_split / type_split / random_split (test sets)
#
# Usage:
#   bash run_cross_split_eval.sh [MODEL]
#
# Examples:
#   bash run_cross_split_eval.sh
#   bash run_cross_split_eval.sh "meta-llama/Llama-3.1-8B-Instruct"
#   bash run_cross_split_eval.sh "Qwen/Qwen2.5-7B-Instruct"

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=2
# MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"

SIMPLE_MODEL="${MODEL//\//-}"

ROOT="/home/yzhang/research/MatProcBench"  # …/MatProcBench
DATA_ROOT="${ROOT}/data/processed"
RAW_FILE="${ROOT}/data/raw_data/MatPROV.jsonl"
RESULTS_DIR="${ROOT}/MatProcAgent/TraceFlow/results/cross_split/${SIMPLE_MODEL}"
EVAL_SCRIPT="${ROOT}/MatProcAgent/eval.py"

TRAIN_FILE="${DATA_ROOT}/additional_split/train.jsonl"

# Inference hyper-parameters (edit here if needed)
TOP_K=8
PLAN_TOKENS=96
ANSWER_TOKENS=48
# EXTRA_FLAGS="--load_in_4bit"   # uncomment for 4-bit quantisation
EXTRA_FLAGS=""

# ── Helpers ────────────────────────────────────────────────────────────────────
run_inference() {
    local SPLIT="$1"
    local TEST_FILE="${DATA_ROOT}/${SPLIT}/test.jsonl"
    local SAVE_FILE="${RESULTS_DIR}/${SPLIT}_traceflow.jsonl"

    echo ""
    echo "======================================================================"
    echo " [Inference] train=additional_split  test=${SPLIT}"
    echo " model=${MODEL}"
    echo "======================================================================"

    if [[ ! -f "${TEST_FILE}" ]]; then
        echo ">>> SKIP — test file not found: ${TEST_FILE}"
        return
    fi

    python -m MatProcAgent.TraceFlow.run_traceflow \
        --train_file  "${TRAIN_FILE}" \
        --data_file   "${TEST_FILE}" \
        --raw_file    "${RAW_FILE}" \
        --save_file   "${SAVE_FILE}" \
        --model       "${MODEL}" \
        --top_k       "${TOP_K}" \
        --plan_max_new_tokens  "${PLAN_TOKENS}" \
        --answer_max_new_tokens "${ANSWER_TOKENS}" \
        ${EXTRA_FLAGS}
}

run_eval() {
    local SPLIT="$1"
    local RESULT_FILE="${RESULTS_DIR}/${SPLIT}_traceflow.jsonl"
    local SAVE_JSON="${RESULTS_DIR}/${SPLIT}_traceflow_eval.json"

    echo ""
    echo "======================================================================"
    echo " [Eval] ${SPLIT}"
    echo "======================================================================"

    if [[ ! -f "${RESULT_FILE}" ]]; then
        echo ">>> SKIP — result file not found: ${RESULT_FILE}"
        return
    fi

    python "${EVAL_SCRIPT}" \
        --result_file "${RESULT_FILE}" \
        --model       "${MODEL}" \
        --mode        mcq \
        --per_task \
        --save_json   "${SAVE_JSON}"
}

# ── Main ───────────────────────────────────────────────────────────────────────
mkdir -p "${RESULTS_DIR}"

echo "======================================================================"
echo " TraceFlow Cross-Split Evaluation"
echo " Train index : additional_split"
echo " Test splits : additional_split, year_split, type_split, random_split"
echo " Model       : ${MODEL}"
echo " Results dir : ${RESULTS_DIR}"
echo "======================================================================"

for SPLIT in additional_split year_split type_split random_split; do
    run_inference "${SPLIT}"
done

echo ""
echo "======================================================================"
echo " All inference done. Running evaluation ..."
echo "======================================================================"

for SPLIT in additional_split year_split type_split random_split; do
    run_eval "${SPLIT}"
done

echo ""
echo "======================================================================"
echo " Cross-split evaluation complete."
echo " Per-task metrics saved in: ${RESULTS_DIR}/"
echo "======================================================================"
