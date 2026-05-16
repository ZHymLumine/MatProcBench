#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
SPLIT="${2:-additional_split}"
SUITE="${3:-full}"
TRAIN_SPLIT="${4:-${TRAIN_SPLIT:-additional_split}}"

SIMPLE_MODEL="${MODEL//\//-}"
DATA_ROOT="${ROOT}/data/processed"
RAW_FILE="${ROOT}/data/raw_data/MatPROV.jsonl"
TRAIN_FILE="${DATA_ROOT}/${TRAIN_SPLIT}/train.jsonl"
DATA_FILE="${DATA_ROOT}/${SPLIT}/test.jsonl"
OUTPUT_DIR="${ROOT}/MatProcAgent/TraceFlow/results/experiments/${SIMPLE_MODEL}/${SPLIT}"

TOP_K="${TOP_K:-8}"
PLAN_TOKENS="${PLAN_TOKENS:-96}"
ANSWER_TOKENS="${ANSWER_TOKENS:-48}"
TOPK_VALUES="${TOPK_VALUES:-1,2,4,8,16}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

echo "======================================================================"
echo " TraceFlow Experiments"
echo " Model      : ${MODEL}"
echo " Train split: ${TRAIN_SPLIT}"
echo " Test split : ${SPLIT}"
echo " Suite      : ${SUITE}"
echo " Output dir : ${OUTPUT_DIR}"
echo "======================================================================"

python -m MatProcAgent.TraceFlow.run_experiments \
  --train_file "${TRAIN_FILE}" \
  --data_file "${DATA_FILE}" \
  --raw_file "${RAW_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --top_k "${TOP_K}" \
  --plan_max_new_tokens "${PLAN_TOKENS}" \
  --answer_max_new_tokens "${ANSWER_TOKENS}" \
  --suite "${SUITE}" \
  --topk_values "${TOPK_VALUES}" \
  ${EXTRA_FLAGS}
