#!/usr/bin/env bash
# ============================================================================
# run_baselines.sh – Run all MatProcBench MCQ baseline methods
# ============================================================================
#
# Supports four split strategies (--split):
#   random_split     – original random 80/10/10 (baseline)
#   year_split       – temporal split: train≤2019 / dev=2020 / test≥2021
#   type_split       – material-type OOD: Battery as test (Combination A)
#   additional_split – double OOD: non-Battery≤2019 train / Battery≥2021 test
#
# Methods: zero_shot / few_shot / cot / rag / graph_rag
#
# Usage:
#   bash run_baselines.sh
#   bash run_baselines.sh --split year_split --model Qwen/Qwen2.5-7B-Instruct
#   bash run_baselines.sh --split type_split --methods "zero_shot cot rag graph_rag"
# ============================================================================

set -e

# For multi-GPU (e.g. 32B/72B), expose all GPUs; device_map="auto" handles sharding.
# To restrict to specific GPUs, set e.g.: export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="HF_TOKEN"

# ── Parse optional CLI overrides ─────────────────────────────────────────────
SPLIT=""
METHODS_OVERRIDE=""
LOAD_IN_4BIT=""

# ── Model ─────────────────────────────────────────────────────────────────────
# MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
# MODEL_NAME="m3rg-iitd/llamat-3-chat"
# MODEL_NAME="m3rg-iitd/llamat-2-chat"
SIMPLE_MODEL_NAME="${MODEL_NAME//\//-}"

# ── Directories ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASELINE_DIR="${SCRIPT_DIR}/.."
EVAL_SCRIPT="${SCRIPT_DIR}/../../eval.py"

# Default processed root (contains all split sub-folders)
PROCESSED_ROOT="${PROCESSED_ROOT:-${SCRIPT_DIR}/../../../data/processed}"
RESULT_ROOT="${RESULT_ROOT:-${BASELINE_DIR}/results}"

# Path to raw MatPROV.jsonl for richer graph construction (graph_rag only)
RAW_FILE="${RAW_FILE:-${SCRIPT_DIR}/../../../data/raw_data/MatPROV.jsonl}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL_NAME="$2";       shift 2 ;;
        --data_root)   DATA_ROOT="$2";        shift 2 ;;
        --result_root) RESULT_ROOT="$2";      shift 2 ;;
        --split)       SPLIT="$2";            shift 2 ;;
        --methods)     METHODS_OVERRIDE="$2"; shift 2 ;;
        --load_in_4bit) LOAD_IN_4BIT="--load_in_4bit"; shift 1 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done


# ── Split selection ───────────────────────────────────────────────────────────
# If not specified via --split, default to running ALL splits sequentially.
ALL_SPLITS="random_split year_split type_split additional_split"
SPLITS_TO_RUN="${SPLIT:-${ALL_SPLITS}}"

# ── Config ────────────────────────────────────────────────────────────────────
N_SHOTS=3
RAG_K=3
SEED=42
COT_MAX_TOKENS=1024
GRAPH_K=3               # graph_rag: number of subgraph neighbours

EMBEDDER="all-mpnet-base-v2"

# Set to 1 to overwrite existing results; 0 to skip
FORCE_RERUN="${FORCE_RERUN:-1}"

# ── Methods to run ────────────────────────────────────────────────────────────
# Default: all five methods.  Override with --methods "zero_shot cot".
DEFAULT_METHODS="zero_shot few_shot rag graph_rag"
# DEFAULT_METHODS=" rag graph_rag"
# DEFAULT_METHODS="zero_shot few_shot"

METHODS="${METHODS_OVERRIDE:-${DEFAULT_METHODS}}"

echo "======================================================================"
echo " MatProcBench Baselines"
echo " Model      : ${MODEL_NAME}"
echo " Splits     : ${SPLITS_TO_RUN}"
echo " Methods    : ${METHODS}"
echo " Result root: ${RESULT_ROOT}"
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
        --result_file  "${RESULT_FILE}" \
        --model        "${MODEL_NAME}" \
        --mode         "mcq" \
        --per_task \
        --save_json    "${SAVE_JSON}"
}

# ── Helper: run one baseline method for a given split ────────────────────────
#   $1 = split name  (e.g. year_split)
#   $2 = method name (e.g. rag)
#   $3+ = extra args passed to run_baselines.py / graph_rag_baseline.py
run_baseline() {
    local SPLIT_NAME="$1"
    local METHOD="$2"
    local EXTRA_ARGS="${@:3}"

    local SPLIT_DATA="${PROCESSED_ROOT}/${SPLIT_NAME}"
    local TRAIN_FILE="${SPLIT_DATA}/train.jsonl"
    local TEST_FILE="${SPLIT_DATA}/test.jsonl"
    local SAVE_DIR="${RESULT_ROOT}/${SIMPLE_MODEL_NAME}/${SPLIT_NAME}/${METHOD}"
    local SAVE_FILE="${SAVE_DIR}/results.jsonl"
    local EMB_CACHE="${SAVE_DIR}/rag_emb_cache.pkl"

    # Validate split folder
    if [[ ! -f "${TEST_FILE}" ]]; then
        echo ""
        echo ">>> [skip] ${SPLIT_NAME}/${METHOD}: test file not found (${TEST_FILE})"
        return
    fi

    # Skip if already done
    if [[ "${FORCE_RERUN}" == "0" && -s "${SAVE_FILE}" ]]; then
        echo ""
        echo ">>> [skip] ${SPLIT_NAME}/${METHOD}: results exist (${SAVE_FILE})"
        return
    fi

    mkdir -p "${SAVE_DIR}"

    echo ""
    echo "======================================================================"
    echo " Split : ${SPLIT_NAME}"
    echo " Method: ${METHOD}"
    echo "======================================================================"

    if [[ "${METHOD}" == "graph_rag" ]]; then
        # graph_rag uses its own entry-point
        local RAW_FILE_ARG=""
        if [[ -f "${RAW_FILE}" ]]; then
            RAW_FILE_ARG="--raw_file ${RAW_FILE}"
        fi
        python "${BASELINE_DIR}/graph_rag_baseline.py" \
            --model      "${MODEL_NAME}" \
            --data_file  "${TEST_FILE}" \
            --save_file  "${SAVE_FILE}" \
            --seed       "${SEED}" \
            --train_file "${TRAIN_FILE}" \
            --graph_k    "${GRAPH_K}" \
            --embedder   "${EMBEDDER}" \
            --emb_cache  "${EMB_CACHE}" \
            ${RAW_FILE_ARG} \
            ${LOAD_IN_4BIT} \
            ${EXTRA_ARGS}
    else
        # All other methods share run_baselines.py
        local METHOD_ARGS=""
        case "${METHOD}" in
            few_shot)
                METHOD_ARGS="--train_file ${TRAIN_FILE} --n_shots ${N_SHOTS}"
                ;;
            cot)
                METHOD_ARGS="--max_new_tokens ${COT_MAX_TOKENS}"
                ;;
            rag)
                METHOD_ARGS="--train_file ${TRAIN_FILE} --rag_k ${RAG_K} \
                             --embedder ${EMBEDDER} --emb_cache ${EMB_CACHE}"
                ;;
        esac

        python "${BASELINE_DIR}/run_baselines.py" \
            --method    "${METHOD}" \
            --model     "${MODEL_NAME}" \
            --data_file "${TEST_FILE}" \
            --save_file "${SAVE_FILE}" \
            --seed      "${SEED}" \
            ${METHOD_ARGS} \
            ${LOAD_IN_4BIT} \
            ${EXTRA_ARGS}
    fi
}

# ── Main loop: iterate splits × methods ──────────────────────────────────────
for SPLIT_NAME in ${SPLITS_TO_RUN}; do
    echo ""
    echo "======================================================================"
    echo " Split: ${SPLIT_NAME}"
    echo "======================================================================"

    for METHOD in ${METHODS}; do
        run_baseline "${SPLIT_NAME}" "${METHOD}"
    done

    # Evaluate all methods for this split
    echo ""
    echo "======================================================================"
    echo " Evaluation  [${SPLIT_NAME}]"
    echo "======================================================================"
    for METHOD in ${METHODS}; do
        run_eval "${RESULT_ROOT}/${SIMPLE_MODEL_NAME}/${SPLIT_NAME}/${METHOD}/results.jsonl"
    done
done

echo ""
echo "======================================================================"
echo " All done. Results under: ${RESULT_ROOT}/${SIMPLE_MODEL_NAME}/"
echo "======================================================================"
