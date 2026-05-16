# TraceFlow

TraceFlow is a fair, training-free, LLM-based agentic baseline for MatProcBench.

It is designed for the actual structure of `MatPROV.jsonl` and uses only
processes that appear in the training split:

- only train-split processes are indexed from `MatPROV.jsonl`
- retrieved train processes are compiled into executable provenance graphs
- symbolic scoring produces task-specific option priors
- Qwen2.5-7B creates a short plan and makes the final decision

This differs from Graph-RAG style methods because the reasoning target is still
structured process reasoning, but the agent is now fair: it cannot directly read
the test process itself. Instead, it reasons over retrieved train-process analogs
plus symbolic signals.

## Method sketch

TraceFlow runs five steps:

1. Train-only process indexing
2. Retrieval of analogous train processes
3. Local graph compilation
4. Symbolic option scoring
5. Qwen2.5-7B planning and final answer selection

## Package layout

TraceFlow is organized as reusable components rather than a single monolithic agent:

- `compiler.py`: compile one raw MatPROV process into an executable `ProcessState`
- `memory.py`: build train-split process memory and analogy retrieval statistics
- `scoring.py`: task-specific symbolic scorers over retrieved process analogs
- `prompting.py`: reusable prompt builders for planning and answer selection
- `agent.py`: thin orchestration layer that assembles the components into `TraceFlowAgent`
- `utils.py`: shared parsing and normalization helpers

Implemented tasks:

- `A1_route_retrieval`
- `A2_missing_step`
- `A3_next_activity`
- `B1_condition_prediction`
- `B2_full_condition_set`
- `C1_tool_selection`
- `D1_process_ordering`

## Usage

```bash
CUDA_VISIBLE_DEVICES=1 python -m MatProcAgent.TraceFlow.run_traceflow \
  --train_file data/processed/additional_split/train.jsonl \
  --data_file data/processed/additional_split/test.jsonl \
  --raw_file data/raw_data/MatPROV.jsonl \
  --save_file MatProcAgent/TraceFlow/results/additional_split_traceflow.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct
```

Then evaluate with the existing evaluator:

```bash
python MatProcAgent/eval.py \
  --result_file MatProcAgent/TraceFlow/results/type_split_traceflow.jsonl \
  --model TraceFlow \
  --mode mcq
```

TraceFlow also supports targeted ablations from the same CLI:

```bash
python -m MatProcAgent.TraceFlow.run_traceflow \
  --train_file data/processed/additional_split/train.jsonl \
  --data_file data/processed/additional_split/test.jsonl \
  --raw_file data/raw_data/MatPROV.jsonl \
  --save_file MatProcAgent/TraceFlow/results/additional_split_llm_only.jsonl \
  --disable_retrieval \
  --disable_symbolic \
  --disable_planning \
  --disable_symbolic_fallback
```

For batch experiments and top-$k$ sweeps:

```bash
bash MatProcAgent/TraceFlow/run_traceflow_experiments.sh \
  "Qwen/Qwen2.5-7B-Instruct" additional_split full additional_split
```

This launcher writes:

- per-configuration `results.jsonl`
- per-configuration `results_eval.json`
- per-configuration `task_family_eval.json`
- experiment-level `summary.json` and `summary.csv`

## Output format

TraceFlow writes JSONL records compatible with `MatProcAgent/eval.py`.

Each record also includes a `traceflow_trace` field with:

- the inferred task
- the retrieved train processes
- the symbolic option priors
- the Qwen-generated reasoning plan
- the raw LLM answer and fallback status
