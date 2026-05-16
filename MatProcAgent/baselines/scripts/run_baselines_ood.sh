#!/usr/bin/env bash
# ============================================================================
# run_baselines_ood.sh – OOD evaluation for MatProcBench MCQ baselines
# ============================================================================
#
# Uses a fixed training split to build prompts / retrieval memory, then
# evaluates the baselines on one or more OOD test splits.
#
# Recommended protocol for the paper:
#   train_split = additional_split
#   test_splits = year_split type_split
#
# Methods:
#   zero_shot  – no train split used; evaluated directly on each test split
#   few_shot   – in-context examples sampled from train_split/train.jsonl
#   cot        – zero-shot chain-of-thought baseline
#   rag        – retrieval memory built from train_split/train.jsonl
#   graph_rag  – graph memory built from train_split/train.jsonl (+ raw MatPROV)
#
# Usage:
#   bash run_baselines_ood.sh
#   bash run_baselines_ood.sh --model Qwen/Qwen2.5-7B-Instruct
#   bash run_baselines_ood.sh --test_splits "additional_split year_split type_split"
#   bash run_baselines_ood.sh --methods "zero_shot few_shot cot rag graph_rag"
# ============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
export HF_TOKEN="${HF_TOKEN:-}"

# ── Defaults ──────────────────────────────────────────────────────────────────
TRAIN_SPLIT="additional_split"
TEST_SPLITS="year_split type_split random_split"
METHODS_OVERRIDE=""
LOAD_IN_4BIT=""

# MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
SIMPLE_MODEL_NAME="${MODEL_NAME//\//-}"

N_SHOTS=3
RAG_K=3
GRAPH_K=3
SEED=42
EMBEDDER="all-mpnet-base-v2"
COT_MAX_TOKENS=1024
FORCE_RERUN="${FORCE_RERUN:-1}"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_DIR="${SCRIPT_DIR}/.."
EVAL_SCRIPT="${SCRIPT_DIR}/../../eval.py"

PROCESSED_ROOT="${PROCESSED_ROOT:-${SCRIPT_DIR}/../../../data/processed}"
RESULT_ROOT="${RESULT_ROOT:-${BASELINE_DIR}/results/ood_additional_train}"
RAW_FILE="${RAW_FILE:-${SCRIPT_DIR}/../../../data/raw_data/MatPROV.jsonl}"

# ── Parse optional CLI overrides ─────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL_NAME="$2";       shift 2 ;;
        --train_split)  TRAIN_SPLIT="$2";      shift 2 ;;
        --test_splits)  TEST_SPLITS="$2";      shift 2 ;;
        --methods)      METHODS_OVERRIDE="$2"; shift 2 ;;
        --result_root)  RESULT_ROOT="$2";      shift 2 ;;
        --data_root|--processed_root)
                        PROCESSED_ROOT="$2";   shift 2 ;;
        --raw_file)     RAW_FILE="$2";         shift 2 ;;
        --n_shots)      N_SHOTS="$2";          shift 2 ;;
        --rag_k)        RAG_K="$2";            shift 2 ;;
        --graph_k)      GRAPH_K="$2";          shift 2 ;;
        --seed)         SEED="$2";             shift 2 ;;
        --embedder)     EMBEDDER="$2";         shift 2 ;;
        --cot_max_tokens) COT_MAX_TOKENS="$2"; shift 2 ;;
        --load_in_4bit) LOAD_IN_4BIT="--load_in_4bit"; shift 1 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

TEST_SPLITS="${TEST_SPLITS//,/ }"
SIMPLE_MODEL_NAME="${MODEL_NAME//\//-}"
DEFAULT_METHODS="zero_shot few_shot rag graph_rag"
METHODS="${METHODS_OVERRIDE:-${DEFAULT_METHODS}}"

TRAIN_FILE="${PROCESSED_ROOT}/${TRAIN_SPLIT}/train.jsonl"

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "Training file not found: ${TRAIN_FILE}" >&2
    exit 1
fi

echo "======================================================================"
echo " MatProcBench Baselines OOD Evaluation"
echo " Model       : ${MODEL_NAME}"
echo " Train split : ${TRAIN_SPLIT}"
echo " Test splits : ${TEST_SPLITS}"
echo " Methods     : ${METHODS}"
echo " Train file  : ${TRAIN_FILE}"
echo " Result root : ${RESULT_ROOT}"
echo "======================================================================"

# ── Helper: evaluate one result file ─────────────────────────────────────────
run_eval() {
    local RESULT_FILE="$1"
    local SAVE_JSON="${RESULT_FILE%.jsonl}_eval.json"
    if [[ ! -f "${RESULT_FILE}" ]]; then
        echo ">>> [eval] Skip (not found): ${RESULT_FILE}"
        return
    fi
    echo ""
    echo ">>> [eval] ${RESULT_FILE}"
    python "${EVAL_SCRIPT}" \
        --result_file "${RESULT_FILE}" \
        --model "${MODEL_NAME}" \
        --mode "mcq" \
        --per_task \
        --save_json "${SAVE_JSON}"
}

# ── Helper: run one baseline method on one test split ────────────────────────
run_baseline() {
    local TEST_SPLIT="$1"
    local METHOD="$2"

    local TEST_FILE="${PROCESSED_ROOT}/${TEST_SPLIT}/test.jsonl"
    local SAVE_DIR="${RESULT_ROOT}/${SIMPLE_MODEL_NAME}/${TRAIN_SPLIT}_to_${TEST_SPLIT}/${METHOD}"
    local SAVE_FILE="${SAVE_DIR}/results.jsonl"
    local EMB_CACHE="${SAVE_DIR}/retrieval_emb_cache.pkl"

    if [[ ! -f "${TEST_FILE}" ]]; then
        echo ""
        echo ">>> [skip] ${TEST_SPLIT}/${METHOD}: test file not found (${TEST_FILE})"
        return
    fi

    if [[ "${FORCE_RERUN}" == "0" && -s "${SAVE_FILE}" ]]; then
        echo ""
        echo ">>> [skip] ${TEST_SPLIT}/${METHOD}: results exist (${SAVE_FILE})"
        return
    fi

    mkdir -p "${SAVE_DIR}"

    echo ""
    echo "======================================================================"
    echo " Train  : ${TRAIN_SPLIT}"
    echo " Test   : ${TEST_SPLIT}"
    echo " Method : ${METHOD}"
    echo "======================================================================"

    if [[ "${METHOD}" == "graph_rag" ]]; then
        local RAW_FILE_ARG=()
        if [[ -f "${RAW_FILE}" ]]; then
            RAW_FILE_ARG=(--raw_file "${RAW_FILE}")
        fi
        python "${BASELINE_DIR}/graph_rag_baseline.py" \
            --model "${MODEL_NAME}" \
            --data_file "${TEST_FILE}" \
            --train_file "${TRAIN_FILE}" \
            --save_file "${SAVE_FILE}" \
            --graph_k "${GRAPH_K}" \
            --embedder "${EMBEDDER}" \
            --emb_cache "${EMB_CACHE}" \
            --seed "${SEED}" \
            "${RAW_FILE_ARG[@]}" \
            ${LOAD_IN_4BIT}
    else
        local METHOD_ARGS=()
        case "${METHOD}" in
            zero_shot)
                ;;
            few_shot)
                METHOD_ARGS=(--train_file "${TRAIN_FILE}" --n_shots "${N_SHOTS}")
                ;;
            cot)
                METHOD_ARGS=(--max_new_tokens "${COT_MAX_TOKENS}")
                ;;
            rag)
                METHOD_ARGS=(
                    --train_file "${TRAIN_FILE}"
                    --rag_k "${RAG_K}"
                    --embedder "${EMBEDDER}"
                    --emb_cache "${EMB_CACHE}"
                )
                ;;
            *)
                echo "Unsupported method for OOD script: ${METHOD}" >&2
                return 1
                ;;
        esac

        python "${BASELINE_DIR}/run_baselines.py" \
            --method "${METHOD}" \
            --model "${MODEL_NAME}" \
            --data_file "${TEST_FILE}" \
            --save_file "${SAVE_FILE}" \
            --seed "${SEED}" \
            "${METHOD_ARGS[@]}" \
            ${LOAD_IN_4BIT}
    fi
}

# ── Main loop ────────────────────────────────────────────────────────────────
for TEST_SPLIT in ${TEST_SPLITS}; do
    echo ""
    echo "======================================================================"
    echo " Test split: ${TEST_SPLIT}"
    echo "======================================================================"

    for METHOD in ${METHODS}; do
        run_baseline "${TEST_SPLIT}" "${METHOD}"
    done

    echo ""
    echo "======================================================================"
    echo " Evaluation [${TRAIN_SPLIT} -> ${TEST_SPLIT}]"
    echo "======================================================================"
    for METHOD in ${METHODS}; do
        run_eval "${RESULT_ROOT}/${SIMPLE_MODEL_NAME}/${TRAIN_SPLIT}_to_${TEST_SPLIT}/${METHOD}/results.jsonl"
    done
done

echo ""
echo "======================================================================"
echo " All done. Results under:"
echo "   ${RESULT_ROOT}/${SIMPLE_MODEL_NAME}/"
echo "======================================================================"
