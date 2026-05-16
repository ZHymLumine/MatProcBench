#!/usr/bin/env bash
# =============================================================================
# run_dual_view.sh — DualView inference + evaluation for MatProcBench
# =============================================================================
#
# Pipeline mode (--mode):
#   dual          dual-view retrieval (text encoder + 2-layer GAT + heuristic)  [DEFAULT]
#   encoder_only  dual-view retrieval only, no scoring
#   symbolic_only heuristic retrieval + symbolic scoring (original ProvMind)
#
# Scoring mode (--scoring_mode, applies to --mode dual only):
#   hybrid        weighted blend of symbolic + neural scoring  [DEFAULT]
#   symbolic      original string-matching scorer
#   neural        embedding NN scorer only
#
# Usage examples:
#   bash run_dual_view.sh
#   bash run_dual_view.sh --split random_split --mode encoder_only
#   bash run_dual_view.sh --split "type_split year_split" --scoring_mode neural
#   bash run_dual_view.sh --split additional_split --infer_only
#   bash run_dual_view.sh --split type_split --eval_only --mode symbolic_only
#   bash run_dual_view.sh --scoring_mode hybrid --scoring_sym_weight 0.6 --scoring_neu_weight 0.4
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

# ── Defaults ──────────────────────────────────────────────────────────────────
SPLITS="additional_split year_split type_split random_split"
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODE="dual"           # dual | encoder_only | symbolic_only
TOP_K=8
PLAN_TOKENS=96
ANSWER_TOKENS=48
TEXT_ENCODER="all-mpnet-base-v2"
ALPHA=0.4
BETA=0.3
GAMMA=0.3
GAT_HIDDEN=256
GAT_HEADS=4
GAT_OUT=256
INFER_ONLY=0
EVAL_ONLY=0
SCORING_MODE="symbolic"   # hybrid | symbolic | neural
SCORING_SYM_W=0.5
SCORING_NEU_W=0.5
EXTRA_FLAGS=""          # e.g. "--load_in_4bit"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split)        SPLITS="$2";       shift 2 ;;
        --model)        MODEL="$2";        shift 2 ;;
        --mode)         MODE="$2";         shift 2 ;;
        --top_k)        TOP_K="$2";        shift 2 ;;
        --text_encoder) TEXT_ENCODER="$2"; shift 2 ;;
        --alpha)        ALPHA="$2";        shift 2 ;;
        --beta)         BETA="$2";         shift 2 ;;
        --gamma)        GAMMA="$2";        shift 2 ;;
        --gat_hidden)   GAT_HIDDEN="$2";   shift 2 ;;
        --gat_heads)    GAT_HEADS="$2";    shift 2 ;;
        --gat_out)      GAT_OUT="$2";      shift 2 ;;
        --infer_only)          INFER_ONLY=1;       shift ;;
        --eval_only)           EVAL_ONLY=1;        shift ;;
        --scoring_mode)        SCORING_MODE="$2";  shift 2 ;;
        --scoring_sym_weight)  SCORING_SYM_W="$2"; shift 2 ;;
        --scoring_neu_weight)  SCORING_NEU_W="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate mode
if [[ "${MODE}" != "dual" && "${MODE}" != "encoder_only" && "${MODE}" != "symbolic_only" ]]; then
    echo "Error: --mode must be one of: dual | encoder_only | symbolic_only"
    exit 1
fi
if [[ "${SCORING_MODE}" != "hybrid" && "${SCORING_MODE}" != "symbolic" && "${SCORING_MODE}" != "neural" ]]; then
    echo "Error: --scoring_mode must be one of: hybrid | symbolic | neural"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="/home/yzhang/research/MatProcBench"                           # …/MatProcBench
DATA_ROOT="${ROOT}/data/processed"
RAW_FILE="${ROOT}/data/raw_data/MatPROV.jsonl"
EVAL_SCRIPT="${ROOT}/MatProcAgent/eval.py"

SIMPLE_MODEL="${MODEL//\//-}"

# Embedding cache shared across splits (keyed by encoder + num_processes inside)
CACHE_DIR="${ROOT}/MatProcAgent/ProvMind/results/.cache/${SIMPLE_MODEL}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "======================================================================"
echo " DualView — MatProcBench"
echo " Splits      : ${SPLITS}"
echo " Mode        : ${MODE}"
echo " Model       : ${MODEL}"
echo " top_k       : ${TOP_K}"
if [[ "${MODE}" != "symbolic_only" ]]; then
    echo " Text encoder: ${TEXT_ENCODER}"
    echo " Fusion      : alpha=${ALPHA}  beta=${BETA}  gamma=${GAMMA}"
    echo " GAT         : hidden=${GAT_HIDDEN}  heads=${GAT_HEADS}  out=${GAT_OUT}"
    echo " Cache dir   : ${CACHE_DIR}"
fi
if [[ "${MODE}" == "dual" ]]; then
    echo " Scoring     : ${SCORING_MODE}  (sym=${SCORING_SYM_W}  neu=${SCORING_NEU_W})"
elif [[ "${MODE}" == "encoder_only" ]]; then
    echo " Scoring     : none  (encoder_only — retrieval quality test)"
else
    echo " Scoring     : symbolic  (symbolic_only — fixed)"
fi
echo "======================================================================"

# ── Helper: build save path ───────────────────────────────────────────────────
results_dir() {
    local SPLIT="$1"
    echo "${ROOT}/MatProcAgent/ProvMind/results/${SIMPLE_MODEL}/${SPLIT}"
}

# Append _score{SCORING_MODE} suffix only for dual mode (where scoring is configurable)
mode_tag() {
    if [[ "${MODE}" == "dual" ]]; then
        echo "${MODE}_score_${SCORING_MODE}"
    else
        echo "${MODE}_score_${SCORING_MODE}"
    fi
}

save_file() {
    local SPLIT="$1"
    echo "$(results_dir "${SPLIT}")/${SPLIT}_dualview_$(mode_tag).jsonl"
}

eval_file() {
    local SPLIT="$1"
    echo "$(results_dir "${SPLIT}")/${SPLIT}_dualview_$(mode_tag)_eval.json"
}

# ── Inference ─────────────────────────────────────────────────────────────────
run_inference() {
    local SPLIT="$1"
    local TRAIN_FILE="${DATA_ROOT}/${SPLIT}/train.jsonl"
    local TEST_FILE="${DATA_ROOT}/${SPLIT}/test.jsonl"
    local SAVE="${SAVE_FILE}"

    echo ""
    echo "----------------------------------------------------------------------"
    echo " [Inference]  split=${SPLIT}  mode=${MODE}"
    echo " train → ${TRAIN_FILE}"
    echo " test  → ${TEST_FILE}"
    echo " save  → ${SAVE}"
    echo "----------------------------------------------------------------------"

    if [[ ! -f "${TRAIN_FILE}" ]]; then
        echo ">>> SKIP — train file not found: ${TRAIN_FILE}"; return
    fi
    if [[ ! -f "${TEST_FILE}" ]]; then
        echo ">>> SKIP — test file not found: ${TEST_FILE}"; return
    fi

    mkdir -p "$(dirname "${SAVE}")"

    if [[ "${MODE}" == "symbolic_only" ]]; then
        python -m MatProcAgent.ProvMind.run_dual_view \
            --data_file             "${TEST_FILE}"   \
            --train_file            "${TRAIN_FILE}"  \
            --raw_file              "${RAW_FILE}"    \
            --save_file             "${SAVE}"        \
            --mode                  symbolic_only    \
            --model                 "${MODEL}"       \
            --top_k                 "${TOP_K}"       \
            --plan_max_new_tokens   "${PLAN_TOKENS}" \
            --answer_max_new_tokens "${ANSWER_TOKENS}" \
            ${EXTRA_FLAGS}
    else
        python -m MatProcAgent.ProvMind.run_dual_view \
            --data_file             "${TEST_FILE}"    \
            --train_file            "${TRAIN_FILE}"   \
            --raw_file              "${RAW_FILE}"     \
            --save_file             "${SAVE}"         \
            --mode                  "${MODE}"         \
            --model                 "${MODEL}"        \
            --top_k                 "${TOP_K}"        \
            --text_encoder          "${TEXT_ENCODER}" \
            --alpha                 "${ALPHA}"        \
            --beta                  "${BETA}"         \
            --gamma                 "${GAMMA}"        \
            --gat_hidden_dim        "${GAT_HIDDEN}"   \
            --gat_heads             "${GAT_HEADS}"    \
            --gat_out_dim           "${GAT_OUT}"      \
            --cache_dir             "${CACHE_DIR}"    \
            --scoring_mode          "${SCORING_MODE}" \
            --scoring_sym_weight    "${SCORING_SYM_W}" \
            --scoring_neu_weight    "${SCORING_NEU_W}" \
            --plan_max_new_tokens   "${PLAN_TOKENS}"  \
            --answer_max_new_tokens "${ANSWER_TOKENS}" \
            ${EXTRA_FLAGS}
    fi

    echo ">>> [Inference] Done → ${SAVE}"
}

# ── Evaluation ────────────────────────────────────────────────────────────────
run_eval() {
    local SPLIT="$1"
    local SAVE="${SAVE_FILE}"
    local EVAL="${EVAL_FILE}"

    echo ""
    echo "----------------------------------------------------------------------"
    echo " [Eval]  split=${SPLIT}  mode=${MODE}"
    echo " result → ${SAVE}"
    echo "----------------------------------------------------------------------"

    if [[ ! -f "${SAVE}" ]]; then
        echo ">>> SKIP — result file not found: ${SAVE}"; return
    fi

    python "${EVAL_SCRIPT}" \
        --result_file "${SAVE}"  \
        --model       "${MODEL}" \
        --mode        mcq        \
        --per_task               \
        --save_json   "${EVAL}"

    echo ">>> [Eval] Done → ${EVAL}"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for SPLIT in ${SPLITS}; do
    SAVE_FILE="$(save_file "${SPLIT}")"
    EVAL_FILE="$(eval_file "${SPLIT}")"

    if [[ "${EVAL_ONLY}" == "0" ]]; then
        run_inference "${SPLIT}"
    fi

    if [[ "${INFER_ONLY}" == "1" ]]; then
        echo ">>> [skip] Eval skipped (--infer_only)."
        continue
    fi

    run_eval "${SPLIT}"
done

echo ""
echo "======================================================================"
echo " All done."
echo " Results under: ${ROOT}/MatProcAgentProvMind/results/${SIMPLE_MODEL}/"
echo " Mode         : ${MODE}"
echo "======================================================================"
