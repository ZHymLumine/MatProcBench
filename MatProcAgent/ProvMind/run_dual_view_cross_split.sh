#!/usr/bin/env bash
# =============================================================================
# run_dual_view_cross_split.sh — DualView cross-split (OOD) evaluation
#
# Train index : additional_split/train.jsonl  (fixed)
# Test splits : additional_split / year_split / type_split / random_split
#
# Pipeline mode (--mode):
#   dual          dual-view retrieval (text + GAT + heuristic)  [DEFAULT]
#   encoder_only  dual-view retrieval only, no scoring
#   symbolic_only heuristic retrieval + symbolic scoring (original TraceFlow)
#
# Scoring mode (--scoring_mode, applies to --mode dual only):
#   hybrid        weighted blend of symbolic + neural scoring  [DEFAULT]
#   symbolic      original string-matching scorer
#   neural        embedding NN scorer only
#
# Usage:
#   bash run_dual_view_cross_split.sh
#   bash run_dual_view_cross_split.sh --mode encoder_only
#   bash run_dual_view_cross_split.sh --scoring_mode neural
#   bash run_dual_view_cross_split.sh --model "meta-llama/Llama-3.1-8B-Instruct" --mode dual
#   bash run_dual_view_cross_split.sh --infer_only
#   bash run_dual_view_cross_split.sh --eval_only --mode symbolic_only
#   bash run_dual_view_cross_split.sh --scoring_mode hybrid --scoring_sym_weight 0.6 --scoring_neu_weight 0.4
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODE="dual"
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
SCORING_MODE="hybrid"   # hybrid | symbolic | neural
SCORING_SYM_W=0.5
SCORING_NEU_W=0.5
EXTRA_FLAGS=""          # e.g. "--load_in_4bit"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
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

if [[ "${MODE}" != "dual" && "${MODE}" != "encoder_only" && "${MODE}" != "symbolic_only" ]]; then
    echo "Error: --mode must be one of: dual | encoder_only | symbolic_only"
    exit 1
fi
if [[ "${SCORING_MODE}" != "hybrid" && "${SCORING_MODE}" != "symbolic" && "${SCORING_MODE}" != "neural" ]]; then
    echo "Error: --scoring_mode must be one of: hybrid | symbolic | neural"
    exit 1
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT="/home/yzhang/research/MatProcBench"
DATA_ROOT="${ROOT}/data/processed"
RAW_FILE="${ROOT}/data/raw_data/MatPROV.jsonl"
EVAL_SCRIPT="${ROOT}/MatProcAgent/eval.py"

SIMPLE_MODEL="${MODEL//\//-}"
TRAIN_FILE="${DATA_ROOT}/additional_split/train.jsonl"
RESULTS_DIR="${ROOT}/MatProcAgent/ProvMind/results/cross_split/${SIMPLE_MODEL}"

# Embedding cache is shared across test splits (train index is fixed)
CACHE_DIR="${ROOT}/MatProcAgent/ProvMind/results/cross_split/.cache/${SIMPLE_MODEL}"

# ── Banner ────────────────────────────────────────────────────────────────────
echo "======================================================================"
echo " DualView Cross-Split (OOD) Evaluation"
echo " Train index : additional_split"
echo " Test splits : additional_split  year_split  type_split  random_split"
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
echo " Results dir : ${RESULTS_DIR}"
echo "======================================================================"

mkdir -p "${RESULTS_DIR}"

# ── Inference ─────────────────────────────────────────────────────────────────
run_inference() {
    local SPLIT="$1"
    local TEST_FILE="${DATA_ROOT}/${SPLIT}/test.jsonl"
    local MODE_TAG="${MODE}"
    MODE_TAG="${MODE}_score_${SCORING_MODE}"
    local SAVE_FILE="${RESULTS_DIR}/${SPLIT}_dualview_${MODE_TAG}.jsonl"

    echo ""
    echo "----------------------------------------------------------------------"
    echo " [Inference] train=additional_split  test=${SPLIT}  mode=${MODE}"
    echo " save → ${SAVE_FILE}"
    echo "----------------------------------------------------------------------"

    if [[ ! -f "${TEST_FILE}" ]]; then
        echo ">>> SKIP — test file not found: ${TEST_FILE}"; return
    fi

    if [[ "${MODE}" == "symbolic_only" ]]; then
        python -m MatProcAgent.ProvMind.run_dual_view \
            --train_file            "${TRAIN_FILE}"    \
            --data_file             "${TEST_FILE}"     \
            --raw_file              "${RAW_FILE}"      \
            --save_file             "${SAVE_FILE}"     \
            --mode                  symbolic_only      \
            --model                 "${MODEL}"         \
            --top_k                 "${TOP_K}"         \
            --plan_max_new_tokens   "${PLAN_TOKENS}"   \
            --answer_max_new_tokens "${ANSWER_TOKENS}" \
            ${EXTRA_FLAGS}
    else
        python -m MatProcAgent.ProvMind.run_dual_view \
            --train_file            "${TRAIN_FILE}"    \
            --data_file             "${TEST_FILE}"     \
            --raw_file              "${RAW_FILE}"      \
            --save_file             "${SAVE_FILE}"     \
            --mode                  "${MODE}"          \
            --model                 "${MODEL}"         \
            --top_k                 "${TOP_K}"         \
            --text_encoder          "${TEXT_ENCODER}"  \
            --alpha                 "${ALPHA}"         \
            --beta                  "${BETA}"          \
            --gamma                 "${GAMMA}"         \
            --gat_hidden_dim        "${GAT_HIDDEN}"    \
            --gat_heads             "${GAT_HEADS}"     \
            --gat_out_dim           "${GAT_OUT}"       \
            --cache_dir             "${CACHE_DIR}"     \
            --scoring_mode          "${SCORING_MODE}"  \
            --scoring_sym_weight    "${SCORING_SYM_W}" \
            --scoring_neu_weight    "${SCORING_NEU_W}" \
            --plan_max_new_tokens   "${PLAN_TOKENS}"   \
            --answer_max_new_tokens "${ANSWER_TOKENS}" \
            ${EXTRA_FLAGS}
    fi

    echo ">>> [Inference] Done → ${SAVE_FILE}"
}

# ── Evaluation ────────────────────────────────────────────────────────────────
run_eval() {
    local SPLIT="$1"
    local MODE_TAG="${MODE}"
    MODE_TAG="${MODE}_score_${SCORING_MODE}"
    # MODE_TAG="${MODE}"
    local RESULT_FILE="${RESULTS_DIR}/${SPLIT}_dualview_${MODE_TAG}.jsonl"
    local SAVE_JSON="${RESULTS_DIR}/${SPLIT}_dualview_${MODE_TAG}_eval.json"

    echo ""
    echo "----------------------------------------------------------------------"
    echo " [Eval] ${SPLIT}  mode=${MODE}"
    echo "----------------------------------------------------------------------"

    if [[ ! -f "${RESULT_FILE}" ]]; then
        echo ">>> SKIP — result file not found: ${RESULT_FILE}"; return
    fi

    python "${EVAL_SCRIPT}" \
        --result_file "${RESULT_FILE}" \
        --model       "${MODEL}"       \
        --mode        mcq              \
        --per_task                     \
        --save_json   "${SAVE_JSON}"

    echo ">>> [Eval] Done → ${SAVE_JSON}"
}

# ── Main ──────────────────────────────────────────────────────────────────────
TEST_SPLITS="additional_split year_split type_split random_split"

if [[ "${EVAL_ONLY}" == "0" ]]; then
    for SPLIT in ${TEST_SPLITS}; do
        run_inference "${SPLIT}"
    done
    echo ""
    echo "======================================================================"
    echo " All inference done."
    echo "======================================================================"
fi

if [[ "${INFER_ONLY}" == "1" ]]; then
    echo ">>> Eval skipped (--infer_only)."
    exit 0
fi

for SPLIT in ${TEST_SPLITS}; do
    run_eval "${SPLIT}"
done

echo ""
echo "======================================================================"
echo " Cross-split evaluation complete."
echo " Mode    : ${MODE}"
echo " Results : ${RESULTS_DIR}/"
echo "======================================================================"
