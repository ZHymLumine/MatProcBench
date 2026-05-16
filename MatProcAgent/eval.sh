#!/usr/bin/env bash
# Evaluate all material synthesis QA results:
#   - MatProcBench baselines (zero_shot / few_shot / cot / rag)
#   - Legacy LLM / LLM-RAG / Graph-CoT results on matprov & matsyn25
# Saves metrics as JSON next to each result file.
#
# Usage: bash eval.sh

set -e

# ── Model & paths ──────────────────────────────────────────────────────────────
# MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
SIMPLE_MODEL_NAME="${MODEL_NAME//\//-}"
HOP="0 1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval.py"

LLM_RESULT_ROOT="${SCRIPT_DIR}/LLM/results"
RAG_RESULT_ROOT="${SCRIPT_DIR}/LLM/rag-results"
GCOT_RESULT_ROOT="${SCRIPT_DIR}/Graph-CoT/results"
BASELINE_RESULT_ROOT="${SCRIPT_DIR}/baselines/results"

# ── Helper: run eval and save metrics ──────────────────────────────────────────
# Args: RESULT_FILE MODE [EXTRA_FLAGS...]
run_eval() {
    local RESULT_FILE="$1"
    local MODE="$2"
    shift 2
    local EXTRA_FLAGS="$@"
    local SAVE_JSON="${RESULT_FILE%.jsonl}_eval.json"

    if [[ ! -f "${RESULT_FILE}" ]]; then
        echo ">>> Skip (not found): ${RESULT_FILE}"
        return
    fi
    echo ""
    echo ">>> Evaluate: $(basename "$(dirname "${RESULT_FILE}")")/$(basename "${RESULT_FILE}") mode=${MODE}"
    python "${EVAL_SCRIPT}" \
        --result_file "${RESULT_FILE}" \
        --model       "${MODEL_NAME}" \
        --mode        "${MODE}" \
        --per_task \
        --save_json   "${SAVE_JSON}" \
        ${EXTRA_FLAGS}
}

# ── MatProcBench Baselines (zero_shot / few_shot / cot / rag) ────────────────
echo ""
echo "======================================================================"
echo " MatProcBench Baselines | Model: ${MODEL_NAME}"
echo "======================================================================"
for METHOD in zero_shot few_shot cot rag; do
    run_eval \
        "${BASELINE_RESULT_ROOT}/${SIMPLE_MODEL_NAME}/${METHOD}/results.jsonl" \
        "mcq"
done

# ── LLM (direct) ──────────────────────────────────────────────────────────────
# echo "======================================================================"
# echo " LLM results | Model: ${MODEL_NAME}"
# echo "======================================================================"
# for DATASET in matprov matsyn25; do
#     run_eval "${LLM_RESULT_ROOT}/${SIMPLE_MODEL_NAME}/${DATASET}/results.jsonl" "free"
#     run_eval "${LLM_RESULT_ROOT}/${SIMPLE_MODEL_NAME}/${DATASET}/results_mcq.jsonl" "mcq"
# done

# ── LLM RAG ───────────────────────────────────────────────────────────────────
# echo ""
# echo "======================================================================"
# echo " LLM RAG results | Model: ${MODEL_NAME} | Hop: ${HOP}"
# echo "======================================================================"
# for DATASET in  matsyn25; do
#     for HOP in ${HOP}; do
#         run_eval "${RAG_RESULT_ROOT}/${SIMPLE_MODEL_NAME}/${DATASET}/results_hop${HOP}.jsonl" "free"
#         run_eval "${RAG_RESULT_ROOT}/${SIMPLE_MODEL_NAME}/${DATASET}/results_hop${HOP}_mcq.jsonl" "mcq"
#     done
# done

# ── Graph-CoT (all variants: tool_mode × plan_mode) ──────────────────────────
echo ""
echo "======================================================================"
echo " Graph-CoT results | Model: ${MODEL_NAME}"
echo "======================================================================"
for DATASET in  matsyn25; do
    for TOOL_MODE in  material; do
        for PLAN_MODE in   llm; do
            VARIANT="${TOOL_MODE}_${PLAN_MODE}"
            GCOT_DIR="${GCOT_RESULT_ROOT}/${SIMPLE_MODEL_NAME}/${DATASET}/${VARIANT}"

            echo ""
            echo "--- Dataset: ${DATASET} | Variant: ${VARIANT} ---"
            run_eval "${GCOT_DIR}/results.jsonl"     "free"
            run_eval "${GCOT_DIR}/results_mcq.jsonl" "mcq"
        done
    done
done

echo ""
echo "======================================================================"
echo " All evaluations done."
echo " Metrics saved as *_eval.json next to each result file."
echo "======================================================================"