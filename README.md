# MatPath

## Project Overview

MatPath is a toolkit for material synthesis reasoning.
It provides:

- Heterogeneous graph construction from materials process data.
- QA dataset construction (free-form and multiple-choice).
- Evaluation workflows for LLM, LLM-RAG, and Graph-based agents.

### Installation

```bash
conda create -n matpath python=3.10
conda activate matpath
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Usage

### 1) Build MCQ Dataset (MatProv)

```bash
python data_utils/build_process_tasks.py \
    --input  data/raw_data/MatPROV.jsonl \
    --output data/processed \
    --seed   42
```

---

## ProvMind

ProvMind is a provenance-aware agent that answers MatProcBench MCQ questions using:

- **Dual-view retrieval** — text encoder (SentenceTransformer) + 2-layer GAT structure + heuristic fusion
- **Hybrid scoring** — weighted blend of symbolic (string-matching) and neural (embedding NN) scoring
- **Reasoning pipeline** — LLM planning → answer generation → symbolic fallback

### 2) Same-split Inference

Run ProvMind on one or more splits (train index = same split):

```bash
# Default: all four splits, dual-view retrieval + symbolic scoring
bash MatProcAgent/ProvMind/run_dual_view.sh

# Single split, hybrid scoring
bash MatProcAgent/ProvMind/run_dual_view.sh \
    --split additional_split --scoring_mode hybrid

# TraceFlow baseline (heuristic retrieval + symbolic scoring only)
bash MatProcAgent/ProvMind/run_traceflow.sh --split additional_split
```

**Retrieval modes** (`--mode`):

| Mode               | Description                                                 |
| ------------------ | ----------------------------------------------------------- |
| `dual` _(default)_ | Text encoder + GAT structure + heuristic fusion             |
| `encoder_only`     | Dual-view retrieval only, no symbolic scoring               |
| `symbolic_only`    | Heuristic retrieval + symbolic scoring (TraceFlow baseline) |

**Scoring modes** (`--scoring_mode`, applies to `--mode dual`):

| Mode                   | Description                                                   |
| ---------------------- | ------------------------------------------------------------- |
| `symbolic` _(default)_ | String-matching + bigram scorer                               |
| `neural`               | Embedding NN scorer                                           |
| `hybrid`               | Weighted blend: `sym_weight * symbolic + neu_weight * neural` |

Results are saved to `MatProcAgent/ProvMind/results/{model}/{split}/`.

### 3) Cross-split OOD Evaluation

Train retrieval index on `additional_split`, evaluate on all four test splits:

```bash
bash MatProcAgent/ProvMind/run_dual_view_cross_split.sh

# With neural scoring
bash MatProcAgent/ProvMind/run_dual_view_cross_split.sh --scoring_mode neural
```

Clean OOD pairs (0% DOI contamination): `additional_split` → `additional_split`, `type_split`, `year_split`.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
