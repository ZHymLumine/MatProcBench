#!/usr/bin/env bash
# ============================================================================
# run_provmind_experiments.sh — ProvMind ablation experiments for MatProcBench
# ============================================================================
#
# Ablation suites:
#   references  Pure LLM / TraceFlow / ProvMind full system
#   retrieval   Decompose retrieval views (text / GAT struct / heuristic)
#   scoring     Scoring mode sweep (symbolic / neural / hybrid variants)
#   modules     Agent module ablations (planning, fallback, symbolic scoring)
#   fusion      Fusion weight sweep (α, β, γ)
#   topk        Top-k retrieval depth sweep
#   full        All of the above (default)
#
# Usage
# -----
#   # Full ablation suite
#   bash run_provmind_experiments.sh
#
#   # Specific suite only
#   bash run_provmind_experiments.sh Qwen/Qwen2.5-7B-Instruct additional_split scoring
#
#   # Cross-split: train on additional_split, test on type_split
#   bash run_provmind_experiments.sh Qwen/Qwen2.5-7B-Instruct type_split full additional_split
#
# Positional args (all optional, override defaults):
#   $1  MODEL        (default: Qwen/Qwen2.5-7B-Instruct)
#   $2  SPLIT        test split (default: additional_split)
#   $3  SUITE        experiment suite (default: full)
#   $4  TRAIN_SPLIT  training split for retrieval index (default: same as SPLIT)
# ============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

# ── Positional args ───────────────────────────────────────────────────────────
MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
SPLIT="${2:-additional_split}"
SUITE="${3:-full}"
TRAIN_SPLIT="${4:-${TRAIN_SPLIT:-${SPLIT}}}"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SIMPLE_MODEL="${MODEL//\//-}"
DATA_ROOT="${ROOT}/data/processed"
RAW_FILE="${ROOT}/data/raw_data/MatPROV.jsonl"
TRAIN_FILE="${DATA_ROOT}/${TRAIN_SPLIT}/train.jsonl"
DATA_FILE="${DATA_ROOT}/${SPLIT}/test.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/results/provmind_experiments/${SIMPLE_MODEL}/${SPLIT}"

# ── Retrieval / scoring hyperparameters (override via env vars) ───────────────
TOP_K="${TOP_K:-8}"
TEXT_ENCODER="${TEXT_ENCODER:-all-mpnet-base-v2}"
ALPHA="${ALPHA:-0.4}"
BETA="${BETA:-0.3}"
GAMMA="${GAMMA:-0.3}"
SCORING_MODE="${SCORING_MODE:-hybrid}"
SCORING_SYM_WEIGHT="${SCORING_SYM_WEIGHT:-0.5}"
SCORING_NEU_WEIGHT="${SCORING_NEU_WEIGHT:-0.5}"
PLAN_TOKENS="${PLAN_TOKENS:-96}"
ANSWER_TOKENS="${ANSWER_TOKENS:-48}"
TOPK_VALUES="${TOPK_VALUES:-1,2,4,8,16}"
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/results/.cache/${SIMPLE_MODEL}}"

EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# ── Cache dir ─────────────────────────────────────────────────────────────────
mkdir -p "${CACHE_DIR}"

echo "======================================================================"
echo " ProvMind Ablation Experiments"
echo " Model        : ${MODEL}"
echo " Train split  : ${TRAIN_SPLIT}"
echo " Test split   : ${SPLIT}"
echo " Suite        : ${SUITE}"
echo " Output dir   : ${OUTPUT_DIR}"
echo " Retrieval    : top_k=${TOP_K}  α=${ALPHA}  β=${BETA}  γ=${GAMMA}"
echo " Scoring      : mode=${SCORING_MODE}  sym=${SCORING_SYM_WEIGHT}  neu=${SCORING_NEU_WEIGHT}"
echo " Cache dir    : ${CACHE_DIR}"
echo "======================================================================"

python -m MatProcAgent.ProvMind.run_provmind_experiments \
  --data_file               "${DATA_FILE}" \
  --train_file              "${TRAIN_FILE}" \
  --raw_file                "${RAW_FILE}" \
  --output_dir              "${OUTPUT_DIR}" \
  --model                   "${MODEL}" \
  --top_k                   "${TOP_K}" \
  --text_encoder            "${TEXT_ENCODER}" \
  --alpha                   "${ALPHA}" \
  --beta                    "${BETA}" \
  --gamma                   "${GAMMA}" \
  --scoring_mode            "${SCORING_MODE}" \
  --scoring_sym_weight      "${SCORING_SYM_WEIGHT}" \
  --scoring_neu_weight      "${SCORING_NEU_WEIGHT}" \
  --plan_max_new_tokens     "${PLAN_TOKENS}" \
  --answer_max_new_tokens   "${ANSWER_TOKENS}" \
  --suite                   "${SUITE}" \
  --topk_values             "${TOPK_VALUES}" \
  --cache_dir               "${CACHE_DIR}" \
  ${EXTRA_FLAGS}

echo ""
echo "======================================================================"
echo " Done. Results under: ${OUTPUT_DIR}/"
echo ""
echo " Suite subdirs:"
for suite_dir in "${OUTPUT_DIR}"/*/; do
  [[ -d "${suite_dir}" ]] || continue
  n=$(find "${suite_dir}" -name "results_eval.json" | wc -l | tr -d ' ')
  echo "   ${suite_dir##*/}/ : ${n} configs evaluated"
done
echo ""
echo " Summary files:"
echo "   ${OUTPUT_DIR}/summary.json"
echo "   ${OUTPUT_DIR}/summary.csv"
echo "======================================================================"
