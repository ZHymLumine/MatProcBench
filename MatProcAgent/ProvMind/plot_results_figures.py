from __future__ import annotations

import json
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MPL_CACHE_DIR = HERE / ".mpl_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import PercentFormatter

MATPROC_AGENT_DIR = HERE.parent
BASELINES_RESULTS = MATPROC_AGENT_DIR / "baselines" / "results"
PROVMIND_RESULTS = HERE / "results"
PROVMIND_TRACEFLOW_RESULTS = HERE / "results_traceflow"
OUTPUT_DIR = HERE / "figures_codex"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = "Qwen-Qwen2.5-7B-Instruct"
DISPLAY_METHODS = [
    "Zero-shot",
    "Few-shot",
    "RAG",
    "GraphRAG",
    "SFT Zero-shot",
    "SFT Few-shot",
    "ProvMind-Hybrid",
]
ANNOTATE_ALL_FIG1_BARS = False
SPLIT_ORDER = ["Additional", "Random", "Type", "Year"]
OOD_SPLITS = ["Additional", "Type", "Year"]
DISPLAY_SPLIT_LABELS = {
    "Additional": "Year+Type",
    "Random": "Random",
    "Type": "Material Type",
    "Year": "Publication Year",
}
SHORT_OOD_LABELS = {
    "Additional": "Year+Type",
    "Type": "Type",
    "Year": "Year",
}
TASK_ID_ORDER = ["D1", "A2", "A3", "C1", "B1", "A1", "B2"]
TASK_ID_TO_NAME = {
    "A1": "Route Retrieval",
    "A2": "Missing Step",
    "A3": "Next Activity",
    "B1": "Condition Prediction",
    "B2": "Full Condition Set",
    "C1": "Tool Selection",
    "D1": "Process Ordering",
}
TASK_ID_TO_SHORT = {
    "A1": "Route Retrieval",
    "A2": "Missing Step",
    "A3": "Next Activity",
    "B1": "Cond. Prediction",
    "B2": "Full Cond. Set",
    "C1": "Tool Selection",
    "D1": "Ordering",
}
TASK_ID_TO_PANEL = {
    "A1": "A1",
    "A2": "A2",
    "A3": "A3",
    "B1": "B1",
    "B2": "B2",
    "C1": "C",
    "D1": "D",
}
TASK_KEY_TO_ID = {
    "A1_route_retrieval": "A1",
    "A2_missing_step": "A2",
    "A3_next_activity": "A3",
    "B1_condition_prediction": "B1",
    "B2_full_condition_set": "B2",
    "C1_tool_selection": "C1",
    "D1_process_ordering": "D1",
}
TASK_FAMILY_ORDER = ["route_operation", "attribute_inference", "causal_ordering"]
CASE_QIDS = {
    "A": "1895",
    "B": "571",
    "C": "20865",
    "D": "12081",
}

COLORS = {
    "bg": "#FCFCFA",
    "panel_bg": "#FFFFFF",
    "panel_border": "#D8D9DE",
    "text": "#1A2433",
    "muted": "#6A7383",
    "grid": "#E8EBF0",
    "axis": "#566377",
    "hybrid": "#799EB9",
    "hybrid_dark": "#5A84A4",
    "neural": "#81D8F8",
    "symbolic": "#FFBE79",
    "baseline": "#7A8598",
    "baseline_light": "#E1E6EE",
    "baseline_mid": "#B3BECC",
    "baseline_dark": "#5D6C82",
    "success": "#66A96D",
    "failure": "#C84C3A",
    "ood": "#F3F5F8",
}

METHOD_COLORS = {
    "Zero-shot": "#FFEFBE",
    "Few-shot": "#F4B183",
    "RAG": "#FFC000",
    "GraphRAG": "#CEC7E6",
    "SFT Zero-shot": "#C5E0B4",
    "SFT Few-shot": "#66A9D9",
    "ProvMind-Hybrid": COLORS["hybrid"],
    "ProvMind-Neural-Only": COLORS["neural"],
    "ProvMind-Symbolic-Only": COLORS["symbolic"],
}

HEATMAP_CMAP = sns.blend_palette(
    ["#EFF3FB", "#C6DCEA", "#799EB9", "#223A5E"],
    as_cmap=True,
)

ABLA_LABELS = {
    "baseline": "ProvMind-Symbolic-Only",
    "no_planning": "ProvMind-Symbolic-Only w/o Planning",
    "no_retrieval": "ProvMind-Symbolic-Only w/o Retrieval",
    "no_symbolic": "ProvMind-Symbolic-Only w/o Symbolic Scoring",
    "llm_decision_only": "ProvMind-Symbolic-Only, LLM Decision-Only",
}

CAPTIONS = {
    "fig1": (
        "Figure 1 | Same-split performance comparison on MatProcBench with "
        "Qwen2.5-7B-Instruct. Accuracy of Zero-shot, Few-shot, RAG, "
        "GraphRAG, SFT Zero-shot, SFT Few-shot, and ProvMind-Hybrid across "
        "the Year+Type, Random, Material Type, and Publication Year splits. "
        "ProvMind-Hybrid remains competitive across all same-split settings "
        "and yields the clearest improvement on the Random split while "
        "preserving consistent gains on the more challenging Material Type "
        "and Publication Year splits."
    ),
    "fig2": (
        "Figure 2 | Clean OOD generalization from Year+Type training. "
        "(A) Overall accuracy on the matched Year+Type split and the clean "
        "OOD Material Type and Publication Year splits. (B) Absolute gain of "
        "ProvMind-Hybrid over SFT Zero-shot on each clean evaluation split. "
        "By excluding the contamination-prone Random split, this figure "
        "isolates generalization under material-type and temporal "
        "distribution shift."
    ),
    "fig3": (
        "Figure 3 | Mechanistic ablations of ProvMind. (A) Ablations within "
        "the symbolic-only variant, isolating the contribution of planning, "
        "retrieval, symbolic scoring, and structured LLM-mediated decision "
        "making. (B) Task-family breakdown of the same symbolic-only variants, "
        "revealing which reasoning skills are most sensitive to removing each "
        "component. (C) View ablation of ProvMind, comparing neural-only, "
        "symbolic-only, and hybrid variants on the Additional, Type, and Year "
        "splits. These panels separate component-level contributions within "
        "symbolic reasoning from higher-level differences between neural, "
        "symbolic, and hybrid formulations of ProvMind."
    ),
    "fig4": (
        "Figure 4 | Task hierarchy and distribution-shift sensitivity of "
        "ProvMind-Hybrid. (A) Per-task accuracy across the Year+Type, "
        "Material Type, Publication Year, and Random splits. (B) Accuracy gap between the Random split "
        "and the average of the other three splits, highlighting tasks that "
        "are most sensitive to distribution shift. (C) Overall task ranking "
        "by non-random average accuracy. Causal ordering remains consistently "
        "easy, whereas route retrieval and full condition-set prediction "
        "remain the dominant bottlenecks under shift."
    ),
    "fig5": (
        "Figure 5 | Qualitative comparison on representative Type-split cases. "
        "Three successful examples show how provenance-aware scoring and "
        "retrieved process analogs correct mistakes made by standard prompting "
        "baselines, while one failure case illustrates a remaining weakness in "
        "long-route discrimination. For each example, we display a compressed "
        "question context, model predictions, and a minimal evidence view "
        "derived from symbolic scores and retrieved process traces."
    ),
}

MPL_STYLE = {
    "figure.facecolor": COLORS["bg"],
    "axes.facecolor": COLORS["panel_bg"],
    "axes.edgecolor": COLORS["axis"],
    "axes.labelcolor": COLORS["axis"],
    "axes.linewidth": 0.9,
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.titlecolor": COLORS["text"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": COLORS["grid"],
    "grid.linewidth": 0.7,
    "grid.alpha": 0.9,
    "xtick.color": COLORS["axis"],
    "ytick.color": COLORS["axis"],
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.prop_cycle": mpl.cycler(color=list(METHOD_COLORS.values())),
    "savefig.facecolor": COLORS["bg"],
}

for key, value in MPL_STYLE.items():
    mpl.rcParams[key] = value

sns.set_theme(style="whitegrid", rc=MPL_STYLE)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def export_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_key_to_label(split_key: str) -> str:
    return split_key.replace("_split", "").capitalize()


def normalize_text(text: str) -> str:
    keep = set("0123456789abcdefghijklmnopqrstuvwxyz .%><=+-/")
    text = text.lower().strip()
    text = "".join(ch if ch in keep else " " for ch in text)
    return " ".join(text.split())


MCQ_PATTERNS = [
    re.compile(r"the\s+answer\s+is\s+([A-D])\b", re.IGNORECASE),
    re.compile(r"\bselect\s+([A-D])\b", re.IGNORECASE),
    re.compile(r"\bchoose\s+([A-D])\b", re.IGNORECASE),
    re.compile(r"\boption\s+([A-D])\b", re.IGNORECASE),
    re.compile(r"\banswer[:\s]+([A-D])\b", re.IGNORECASE),
    re.compile(r"\b([A-D])\s+is\s+(?:correct|right)\b", re.IGNORECASE),
]


def choices_text_match(raw: str, choices: list[Any]) -> str:
    if not choices:
        return ""
    norm_pred = normalize_text(raw)
    if not norm_pred:
        return ""

    for idx, choice in enumerate(choices):
        if normalize_text(str(choice)) == norm_pred:
            return chr(65 + idx)

    alnum = re.compile(r"[a-z0-9]", re.IGNORECASE)
    numeric_op = re.compile(r"^[><=!~]+$")
    best_letter, best_score = "", 0
    for idx, choice in enumerate(choices):
        norm_choice = normalize_text(str(choice))
        if not norm_choice or norm_pred not in norm_choice:
            continue
        pos = norm_choice.find(norm_pred)
        prefix = norm_choice[:pos].strip()
        suffix = norm_choice[pos + len(norm_pred) :].strip()
        if alnum.search(prefix) or alnum.search(suffix):
            continue
        if prefix and numeric_op.match(prefix):
            continue
        if len(norm_pred) > best_score:
            best_letter = chr(65 + idx)
            best_score = len(norm_pred)
    if best_letter:
        return best_letter

    best_letter, best_score = "", 0
    for idx, choice in enumerate(choices):
        norm_choice = normalize_text(str(choice))
        if norm_choice in norm_pred and len(norm_choice) > best_score:
            if len(norm_choice) >= max(3, int(len(norm_pred) * 0.6)):
                best_letter = chr(65 + idx)
                best_score = len(norm_choice)
    return best_letter


def extract_mcq_letter(text: Any, choices: list[Any] | None = None) -> str:
    raw = "" if text is None else str(text).strip()
    if not raw:
        return ""
    match = re.match(r"^([A-D])[\s\.\)\:\,\-]", raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if re.match(r"^([A-D])$", raw, re.IGNORECASE):
        return raw[0].upper()
    for pattern in MCQ_PATTERNS:
        match = pattern.search(raw)
        if match:
            return match.group(1).upper()
    letter = choices_text_match(raw, choices or [])
    if letter:
        return letter
    alpha_only = re.sub(r"[^A-Za-z]", "", raw)
    if len(alpha_only) == 1 and alpha_only.upper() in "ABCD":
        return alpha_only.upper()
    return ""


def gold_letter(record: dict[str, Any]) -> str:
    gold = str(record.get("gt_answer", "")).strip()
    if len(gold) == 1 and gold.upper() in "ABCD":
        return gold.upper()
    answer_index = record.get("answer_index")
    if isinstance(answer_index, int) and answer_index >= 0:
        return chr(65 + answer_index)
    choices = record.get("choices", [])
    norm_gold = normalize_text(gold)
    for idx, choice in enumerate(choices):
        if normalize_text(str(choice)) == norm_gold:
            return chr(65 + idx)
    return gold.upper()


def compute_accuracy_from_jsonl(records: list[dict[str, Any]]) -> tuple[float, int, int]:
    correct = 0
    for record in records:
        pred = extract_mcq_letter(record.get("model_answer", ""), record.get("choices", []))
        gold = gold_letter(record)
        if pred == gold:
            correct += 1
    total = len(records)
    return (100.0 * correct / total) if total else 0.0, correct, total


def metric_from_eval(data: dict[str, Any]) -> tuple[float, int | None, int | None]:
    acc = data.get("MCQ_accuracy", data.get("EM"))
    correct = data.get("MCQ_correct")
    total = data.get("MCQ_total", data.get("total"))
    if correct is None and acc is not None and total:
        correct = int(round(acc * total))
    return float(acc) * 100.0, correct, total


def collect_same_split_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method_dirs = {
        "Zero-shot": "zero_shot",
        "Few-shot": "few_shot",
        "RAG": "rag",
        "GraphRAG": "graph_rag",
        "SFT Zero-shot": "sft_zero_shot",
        "SFT Few-shot": "sft_few_shot",
    }
    for split_key in ["additional_split", "random_split", "type_split", "year_split"]:
        split_label = split_key_to_label(split_key)
        for method_name, method_dir in method_dirs.items():
            eval_path = BASELINES_RESULTS / MODEL_DIR / split_key / method_dir / "results_eval.json"
            accuracy, correct, total = metric_from_eval(load_json(eval_path))
            rows.append(
                {
                    "family": "same_split",
                    "variant": method_name,
                    "split": split_label,
                    "train_split": split_label,
                    "test_split": split_label,
                    "setting": "same_split",
                    "task": None,
                    "task_family": None,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "qid": None,
                }
            )
        hybrid_path = PROVMIND_RESULTS / MODEL_DIR / split_key / f"{split_key}_dualview_dual_eval.json"
        accuracy, correct, total = metric_from_eval(load_json(hybrid_path))
        rows.append(
            {
                "family": "same_split",
                "variant": "ProvMind-Hybrid",
                "split": split_label,
                "train_split": split_label,
                "test_split": split_label,
                "setting": "same_split",
                "task": None,
                "task_family": None,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "qid": None,
            }
        )
    return rows


def collect_cross_split_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    baseline_dir = BASELINES_RESULTS / "cross_split" / MODEL_DIR / "additional_split" / "zero_shot" / "zero_shot"
    for test_key in ["additional_split", "type_split", "year_split"]:
        records = load_jsonl(baseline_dir / test_key / "results.jsonl")
        accuracy, correct, total = compute_accuracy_from_jsonl(records)
        rows.append(
            {
                "family": "clean_ood",
                "variant": "SFT Zero-shot",
                "split": split_key_to_label(test_key),
                "train_split": "Additional",
                "test_split": split_key_to_label(test_key),
                "setting": "cross_split",
                "task": None,
                "task_family": None,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "qid": None,
            }
        )
        hybrid_eval = PROVMIND_RESULTS / "cross_split" / MODEL_DIR / f"{test_key}_dualview_dual_eval.json"
        accuracy, correct, total = metric_from_eval(load_json(hybrid_eval))
        rows.append(
            {
                "family": "clean_ood",
                "variant": "ProvMind-Hybrid",
                "split": split_key_to_label(test_key),
                "train_split": "Additional",
                "test_split": split_key_to_label(test_key),
                "setting": "cross_split",
                "task": None,
                "task_family": None,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "qid": None,
            }
        )
    return rows


def collect_symbolic_ablation_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = PROVMIND_TRACEFLOW_RESULTS / "experiments" / MODEL_DIR / "additional_split" / "ablations"
    for key in ["baseline", "no_planning", "no_retrieval", "no_symbolic", "llm_decision_only"]:
        overall = load_json(root / key / "results_eval.json")
        accuracy, correct, total = metric_from_eval(overall)
        rows.append(
            {
                "family": "symbolic_ablation",
                "variant": ABLA_LABELS[key],
                "split": "Additional",
                "train_split": "Additional",
                "test_split": "Additional",
                "setting": "symbolic_ablation",
                "task": None,
                "task_family": None,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "qid": None,
            }
        )
        task_family = load_json(root / key / "task_family_eval.json")
        for family_name in TASK_FAMILY_ORDER:
            family_metrics = task_family[family_name]
            rows.append(
                {
                    "family": "symbolic_ablation",
                    "variant": ABLA_LABELS[key],
                    "split": "Additional",
                    "train_split": "Additional",
                    "test_split": "Additional",
                    "setting": "task_family",
                    "task": None,
                    "task_family": family_name,
                    "accuracy": family_metrics["MCQ_accuracy"] * 100.0,
                    "correct": family_metrics["MCQ_correct"],
                    "total": family_metrics["MCQ_total"],
                    "qid": None,
                }
            )
    return rows


def collect_view_ablation_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method_map = {
        "encoder_only": "ProvMind-Neural-Only",
        "symbolic_only": "ProvMind-Symbolic-Only",
        "dual": "ProvMind-Hybrid",
    }
    root = PROVMIND_RESULTS / "cross_split" / MODEL_DIR
    for split_key in ["additional_split", "type_split", "year_split"]:
        split_label = split_key_to_label(split_key)
        for key, variant in method_map.items():
            accuracy, correct, total = metric_from_eval(load_json(root / f"{split_key}_dualview_{key}_eval.json"))
            rows.append(
                {
                    "family": "view_ablation",
                    "variant": variant,
                    "split": split_label,
                    "train_split": "Additional",
                    "test_split": split_label,
                    "setting": "view_ablation",
                    "task": None,
                    "task_family": None,
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": total,
                    "qid": None,
                }
            )
    return rows


def collect_task_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_key in ["additional_split", "random_split", "type_split", "year_split"]:
        split_label = split_key_to_label(split_key)
        per_task = load_json(PROVMIND_RESULTS / MODEL_DIR / split_key / f"{split_key}_dualview_dual_eval.json")["per_task"]
        for task_key, metrics in per_task.items():
            rows.append(
                {
                    "family": "hybrid_per_task",
                    "variant": "ProvMind-Hybrid",
                    "split": split_label,
                    "train_split": split_label,
                    "test_split": split_label,
                    "setting": "per_task",
                    "task": TASK_KEY_TO_ID[task_key],
                    "task_family": None,
                    "accuracy": metrics["MCQ_accuracy"] * 100.0,
                    "correct": metrics["MCQ_correct"],
                    "total": metrics["MCQ_total"],
                    "qid": None,
                }
            )
    return rows


def collect_case_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = {
        "Few-shot": BASELINES_RESULTS / MODEL_DIR / "type_split" / "few_shot" / "results.jsonl",
        "ProvMind-Neural-Only": PROVMIND_RESULTS / MODEL_DIR / "type_split" / "type_split_dualview_encoder_only.jsonl",
        "ProvMind-Symbolic-Only": PROVMIND_RESULTS / MODEL_DIR / "type_split" / "type_split_dualview_symbolic_only.jsonl",
        "ProvMind-Hybrid": PROVMIND_RESULTS / MODEL_DIR / "type_split" / "type_split_dualview_dual.jsonl",
    }
    loaded = {name: {row["qid"]: row for row in load_jsonl(path)} for name, path in paths.items()}
    for panel, qid in CASE_QIDS.items():
        hybrid = loaded["ProvMind-Hybrid"][qid]
        trace = hybrid.get("traceflow_trace", {})
        symbolic = trace.get("symbolic", {})
        rows.append(
            {
                "panel": panel,
                "qid": qid,
                "task": TASK_KEY_TO_ID[hybrid["task"]],
                "question": hybrid["question"],
                "choices": hybrid.get("choices", []),
                "gt": hybrid["gt_answer"],
                "Few-shot": loaded["Few-shot"][qid]["model_answer"],
                "ProvMind-Neural-Only": loaded["ProvMind-Neural-Only"][qid]["model_answer"],
                "ProvMind-Symbolic-Only": loaded["ProvMind-Symbolic-Only"][qid]["model_answer"],
                "ProvMind-Hybrid": hybrid["model_answer"],
                "retrieved_processes": [p.get("doi", "") for p in trace.get("retrieved_processes", [])[:2]],
                "visible_edges": symbolic.get("visible_edges", []),
                "symbolic_scores": symbolic.get("option_scores", {}),
                "success": panel != "D",
            }
        )
    return rows


def export_captions() -> None:
    lines = ["# Figure Captions", ""]
    for key in ["fig1", "fig2", "fig3", "fig4", "fig5"]:
        lines.append(f"## {key.upper()}")
        lines.append(CAPTIONS[key])
        lines.append("")
    (OUTPUT_DIR / "figure_captions.md").write_text("\n".join(lines))


def display_split_label(split: str) -> str:
    return DISPLAY_SPLIT_LABELS.get(split, split)


def figure_header(fig: plt.Figure, fig_label: str, subtitle: str) -> None:
    return


def panel_title(ax: plt.Axes, label: str, title: str) -> None:
    full = label if not title else f"{label}  {title}"
    ax.set_title(full, loc="left", pad=10, fontsize=13.0, fontweight="bold", color=COLORS["text"])


def panel_bg(ax: plt.Axes, rounded: bool = False) -> None:
    ax.set_facecolor(COLORS["panel_bg"])
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    if rounded:
        ax.patch.set_alpha(0)
        patch = FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            transform=ax.transAxes,
            linewidth=1.0,
            edgecolor=COLORS["panel_border"],
            facecolor=COLORS["panel_bg"],
            zorder=-10,
            clip_on=False,
        )
        ax.add_patch(patch)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300)
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    plt.close(fig)


def best_displayed_baseline(df: pd.DataFrame) -> pd.DataFrame:
    baseline_df = df[df["variant"].isin(DISPLAY_METHODS[:-1])].copy()
    idx = baseline_df.groupby("split")["accuracy"].idxmax()
    best = baseline_df.loc[idx, ["split", "variant", "accuracy"]].rename(
        columns={"variant": "best_baseline", "accuracy": "baseline_accuracy"}
    )
    hybrid = (
        df[df["variant"] == "ProvMind-Hybrid"][["split", "accuracy"]]
        .rename(columns={"accuracy": "hybrid_accuracy"})
        .copy()
    )
    merged = best.merge(hybrid, on="split", how="left")
    merged["delta"] = merged["hybrid_accuracy"] - merged["baseline_accuracy"]
    merged["split"] = pd.Categorical(merged["split"], SPLIT_ORDER, ordered=True)
    return merged.sort_values("split")


def short_ablation_label(label: str) -> str:
    return (
        label.replace("ProvMind-Symbolic-Only", "PSO")
        .replace(" w/o ", "\nwithout ")
        .replace(", ", "\n")
    )


def annotate_bars(ax: plt.Axes, fmt: str = "{:.1f}", color: str = COLORS["text"], dy: float = 0.5) -> None:
    for patch in ax.patches:
        if not isinstance(patch, Rectangle):
            continue
        height = patch.get_height()
        if math.isnan(height):
            continue
        x = patch.get_x() + patch.get_width() / 2
        ax.text(x, height + dy, fmt.format(height), ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")


def draw_fig1(metrics: pd.DataFrame) -> None:
    df = metrics[(metrics["family"] == "same_split") & (metrics["variant"].isin(DISPLAY_METHODS))].copy()
    df["split"] = pd.Categorical(df["split"], SPLIT_ORDER, ordered=True)
    df["variant"] = pd.Categorical(df["variant"], DISPLAY_METHODS, ordered=True)
    fig = plt.figure(figsize=(13.6, 7.9))
    gs = fig.add_gridspec(1, 1, left=0.08, right=0.985, bottom=0.11, top=0.93)
    ax1 = fig.add_subplot(gs[0, 0])
    panel_bg(ax1, rounded=True)

    x = np.arange(len(SPLIT_ORDER))
    width = 0.1
    offsets = np.linspace(-3, 3, len(DISPLAY_METHODS)) * width
    for method, offset in zip(DISPLAY_METHODS, offsets):
        sub = df[df["variant"] == method].sort_values("split")
        values = sub["accuracy"].to_numpy()
        ax1.bar(
            x + offset,
            values,
            width=width * 0.92,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            label=method,
        )
        if ANNOTATE_ALL_FIG1_BARS:
            for xi, yi in zip(x + offset, values):
                ax1.text(
                    xi,
                    yi + 0.8,
                    f"{yi:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    rotation=90,
                    color=COLORS["axis"],
                )
        elif method == "ProvMind-Hybrid":
            for xi, yi in zip(x + offset, values):
                ax1.text(xi, yi + 1.0, f"{yi:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLORS["text"])

    ax1.set_xticks(x)
    ax1.set_xticklabels([display_split_label(s) for s in SPLIT_ORDER], fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(
        ncol=4,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.005),
        fontsize=9.6,
        handlelength=1.0,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    save_figure(fig, "fig1_same_split_overview")


def draw_fig2(metrics: pd.DataFrame) -> None:
    df = metrics[(metrics["family"] == "clean_ood") & (metrics["variant"].isin(["SFT Zero-shot", "ProvMind-Hybrid"]))].copy()
    df["split"] = pd.Categorical(df["split"], OOD_SPLITS, ordered=True)
    df = df.sort_values(["variant", "split"])
    delta_df = (
        df.pivot(index="split", columns="variant", values="accuracy")
        .reset_index()
        .assign(delta=lambda x: x["ProvMind-Hybrid"] - x["SFT Zero-shot"])
    )

    fig = plt.figure(figsize=(14.2, 6.8))
    gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, bottom=0.14, top=0.92, wspace=0.26, width_ratios=[1.02, 1.02])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    for ax in (ax1, ax2):
        panel_bg(ax, rounded=True)

    panel_title(ax1, "A", "Overall accuracy")
    panel_title(ax2, "B", "Hybrid gain")

    x = np.arange(len(OOD_SPLITS))
    width = 0.28
    methods = ["SFT Zero-shot", "ProvMind-Hybrid"]
    for idx, method in enumerate(methods):
        sub = df[df["variant"] == method].sort_values("split")
        ax1.bar(
            x + (idx - 0.5) * width,
            sub["accuracy"].to_numpy(),
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.8,
            label=method,
            zorder=3,
        )
    annotate_bars(ax1, dy=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Year+Type", "Material Type", "Publication Year"], fontweight="bold")
    ax1.set_ylim(0, 70)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.01), ncol=2, fontsize=9.5, handlelength=1.6, columnspacing=1.2)

    split_short_labels = [SHORT_OOD_LABELS[split] for split in delta_df["split"]]
    colors = [COLORS["hybrid"] if val >= 0 else COLORS["failure"] for val in delta_df["delta"]]
    ax2.barh(split_short_labels, delta_df["delta"], color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    d_min = float(delta_df["delta"].min())
    d_max = float(delta_df["delta"].max())
    margin = (d_max - d_min) * 0.18 + 1.0
    ax2.set_xlim(0, d_max + margin)
    ax2.axvline(0, color=COLORS["axis"], lw=1.2)
    for y, val in zip(split_short_labels, delta_df["delta"]):
        offset = 0.4 if val >= 0 else -0.4
        ha = "left" if val >= 0 else "right"
        ax2.text(val + offset, y, f"{val:+.1f}pp", va="center", ha=ha,
                 fontsize=9.5, fontweight="bold", color=COLORS["text"])
    ax2.tick_params(axis="y", length=0, pad=10, labelsize=9.6)
    for tick in ax2.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_color(COLORS["axis"])
    ax2.set_xlabel("Absolute gain (pp)", labelpad=6)
    ax2.set_ylabel("")
    save_figure(fig, "fig2_clean_ood_generalization")


def draw_fig3(metrics: pd.DataFrame) -> None:
    symbolic = metrics[(metrics["family"] == "symbolic_ablation") & (metrics["setting"] == "symbolic_ablation")].copy()
    symbolic = symbolic.sort_values("accuracy", ascending=False)
    symbolic_task = metrics[(metrics["family"] == "symbolic_ablation") & (metrics["setting"] == "task_family")].copy()
    view = metrics[metrics["family"] == "view_ablation"].copy()
    view["split"] = pd.Categorical(view["split"], OOD_SPLITS, ordered=True)
    view["variant"] = pd.Categorical(
        view["variant"],
        ["ProvMind-Neural-Only", "ProvMind-Symbolic-Only", "ProvMind-Hybrid"],
        ordered=True,
    )

    fig = plt.figure(figsize=(16, 7.8))
    gs = fig.add_gridspec(1, 3, left=0.13, right=0.985, bottom=0.14, top=0.92, wspace=0.22, width_ratios=[0.95, 1.05, 0.95])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    for ax in (ax1, ax2, ax3):
        panel_bg(ax, rounded=True)

    panel_title(ax1, "A", "Symbolic-stack ablations")
    panel_title(ax2, "B", "Task-family sensitivity")
    panel_title(ax3, "C", "View ablation on clean OOD")

    compact_symbolic_labels = {
        "ProvMind-Symbolic-Only": "Symbolic-Only",
        "ProvMind-Symbolic-Only w/o Planning": "w/o Planning",
        "ProvMind-Symbolic-Only w/o Retrieval": "w/o Retrieval",
        "ProvMind-Symbolic-Only w/o Symbolic Scoring": "w/o Sym. Scoring",
        "ProvMind-Symbolic-Only, LLM Decision-Only": "LLM Decision Only",
    }
    y = np.arange(len(symbolic))
    colors = [COLORS["symbolic"]] + [COLORS["baseline_mid"]] * (len(symbolic) - 1)
    ax1.barh(y, symbolic["accuracy"], color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([compact_symbolic_labels[label] for label in symbolic["variant"]], fontsize=9.2, fontweight="bold")
    ax1.tick_params(axis="y", pad=8)
    ax1.invert_yaxis()
    ax1.set_xlim(44, 56)
    ax1.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    for yi, val in zip(y, symbolic["accuracy"]):
        ax1.text(val + 0.12, yi, f"{val:.2f}", va="center", fontsize=9, fontweight="bold", color=COLORS["text"])
    ax1.set_xlabel("Accuracy (%)")

    heatmap_df = symbolic_task.pivot(index="task_family", columns="variant", values="accuracy").loc[TASK_FAMILY_ORDER, [ABLA_LABELS[k] for k in ["baseline", "no_planning", "no_retrieval", "no_symbolic", "llm_decision_only"]]]
    sns.heatmap(
        heatmap_df,
        ax=ax2,
        cmap=HEATMAP_CMAP,
        annot=True,
        fmt=".1f",
        cbar=False,
        linewidths=3,
        linecolor=COLORS["bg"],
        annot_kws={"fontsize": 9.5, "fontweight": "bold"},
    )
    heatmap_labels = {
        "ProvMind-Symbolic-Only": "Symbolic-Only",
        "ProvMind-Symbolic-Only w/o Planning": "w/o Planning",
        "ProvMind-Symbolic-Only w/o Retrieval": "w/o Retrieval",
        "ProvMind-Symbolic-Only w/o Symbolic Scoring": "w/o Sym. Scoring",
        "ProvMind-Symbolic-Only, LLM Decision-Only": "LLM Decision Only",
    }
    compact_task_family = {
        "route_operation": "Route",
        "attribute_inference": "Attribute",
        "causal_ordering": "Ordering",
    }
    ax2.set_yticklabels([compact_task_family[label] for label in heatmap_df.index], rotation=0, fontsize=8.8, fontweight="bold")
    ax2.set_xticklabels([heatmap_labels[label] for label in heatmap_df.columns], rotation=0, fontsize=8.2)
    ax2.tick_params(length=0)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    x = np.arange(len(OOD_SPLITS))
    width = 0.22
    variants = ["ProvMind-Neural-Only", "ProvMind-Symbolic-Only", "ProvMind-Hybrid"]
    for idx, variant in enumerate(variants):
        sub = view[view["variant"] == variant].sort_values("split")
        ax3.bar(
            x + (idx - 1) * width,
            sub["accuracy"].to_numpy(),
            width=width,
            color=METHOD_COLORS[variant],
            edgecolor="white",
            linewidth=0.8,
            label=variant.replace("ProvMind-", ""),
            zorder=3,
        )
    annotate_bars(ax3, dy=0.35)
    ax3.set_xticks(x)
    ax3.set_xticklabels(["Year+Type", "Material Type", "Pub. Year"], fontweight="bold")
    ax3.set_ylim(40, 60)
    ax3.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax3.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=8.8, handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
    ax3.set_ylabel("")
    save_figure(fig, "fig3_mechanistic_ablations")


def draw_fig4(metrics: pd.DataFrame) -> None:
    task_df = metrics[metrics["family"] == "hybrid_per_task"].copy()
    task_df["split"] = pd.Categorical(task_df["split"], SPLIT_ORDER, ordered=True)
    pivot = task_df.pivot(index="task", columns="split", values="accuracy")
    non_random_mean = pivot[["Additional", "Type", "Year"]].mean(axis=1)
    ranked_tasks = non_random_mean.sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[ranked_tasks, SPLIT_ORDER]

    gap_df = pd.DataFrame(
        {
            "task": ranked_tasks,
            "gap": pivot["Random"] - pivot[["Additional", "Type", "Year"]].mean(axis=1),
            "mean_non_random": non_random_mean.loc[ranked_tasks].to_numpy(),
        }
    )

    fig = plt.figure(figsize=(16, 8.0))
    gs = fig.add_gridspec(2, 2, left=0.13, right=0.985, bottom=0.12, top=0.92, wspace=0.10, hspace=0.30, width_ratios=[1.10, 0.90], height_ratios=[1.0, 0.82])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    for ax in (ax1, ax2, ax3):
        panel_bg(ax, rounded=True)

    panel_title(ax1, "A", "Per-task accuracy heatmap")
    panel_title(ax2, "B", "Random-split boost")
    panel_title(ax3, "C", "Task difficulty ranking")

    display_index = [f"{TASK_ID_TO_PANEL[task]}  {TASK_ID_TO_SHORT[task]}" for task in pivot.index]
    display_columns = [display_split_label(s) for s in ["Additional", "Random", "Type", "Year"]]
    heat_df = pivot.copy()
    heat_df.index = display_index
    heat_df.columns = display_columns
    sns.heatmap(
        heat_df,
        ax=ax1,
        cmap=HEATMAP_CMAP,
        annot=True,
        fmt=".1f",
        cbar=False,
        linewidths=3,
        linecolor=COLORS["bg"],
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )
    ax1.tick_params(length=0)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_xticklabels(display_columns, rotation=0, fontsize=10.5, fontweight="bold")
    ax1.set_yticklabels(display_index, rotation=0, fontsize=9.8, fontweight="bold")

    gap_df = gap_df.sort_values("gap", ascending=False)
    gap_positions = np.arange(len(gap_df))
    gap_colors = [COLORS["hybrid"] if v >= 0 else COLORS["failure"] for v in gap_df["gap"]]
    ax2.barh(gap_positions, gap_df["gap"], color=gap_colors, edgecolor="white", linewidth=0.8, zorder=3)
    gap_min = float(gap_df["gap"].min())
    gap_max = float(gap_df["gap"].max())
    gap_margin = max(3.0, (gap_max - gap_min) * 0.10 + 1.4)
    ax2.set_xlim(min(-3.0, gap_min - 1.2), gap_max + gap_margin)
    ax2.axvline(0, color=COLORS["axis"], lw=1.2)
    for pos, val in zip(gap_positions, gap_df["gap"]):
        offset = 0.4 if val >= 0 else -0.4
        ha = "left" if val >= 0 else "right"
        ax2.text(val + offset, pos, f"{val:+.1f}pp", va="center", ha=ha,
                 fontsize=9.5, fontweight="bold", color=COLORS["text"])
    ax2.set_xlabel("Gain over non-random mean (pp)", labelpad=6)
    ax2.set_ylabel("")
    ax2.set_yticks(gap_positions)
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", length=0)
    x_gap_text = max(0.4, ax2.get_xlim()[0] + 0.6)
    for pos, task in zip(gap_positions, gap_df["task"]):
        ax2.text(x_gap_text, pos, TASK_ID_TO_PANEL[task], va="center", ha="left", fontsize=10, fontweight="bold", color=COLORS["axis"])

    rank_df = gap_df.sort_values("mean_non_random", ascending=False).reset_index(drop=True)
    colors = sns.color_palette([COLORS["hybrid"]], n_colors=len(rank_df))
    colors = [mpl.colors.to_hex((0.12, 0.5 + 0.35 * (1 - i / max(1, len(rank_df) - 1)), 0.48)) for i in range(len(rank_df))]
    rank_positions = np.arange(len(rank_df))
    ax3.barh(rank_positions, rank_df["mean_non_random"], color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for pos, val in zip(rank_positions, rank_df["mean_non_random"]):
        ax3.text(val + 0.35, pos, f"{val:.1f}", va="center", fontsize=9.5, fontweight="bold", color=COLORS["text"])
    ax3.set_ylabel("")
    ax3.set_xlabel("Mean accuracy on non-random splits", labelpad=6)
    ax3.set_yticks(rank_positions)
    ax3.set_yticklabels([])
    ax3.tick_params(axis="y", length=0)
    x_rank_text = ax3.get_xlim()[0] + 1.5
    for pos, task in zip(rank_positions, rank_df["task"]):
        ax3.text(x_rank_text, pos, TASK_ID_TO_PANEL[task], va="center", ha="left", fontsize=10, fontweight="bold", color=COLORS["axis"])
    save_figure(fig, "fig4_task_hierarchy_shift")


def truncate_wrapped(text: str, width: int, max_lines: int) -> str:
    wrapped = textwrap.wrap(text, width=width) or [""]
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)
    clipped = wrapped[:max_lines]
    clipped[-1] = clipped[-1][: max(0, len(clipped[-1]) - 1)] + "…"
    return "\n".join(clipped)


def shorten_question(question: str, lines: int = 5, width: int = 54) -> str:
    parts = [line.strip() for line in question.splitlines() if line.strip()]
    parts = parts[:lines]
    wrapped = []
    for line in parts:
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    wrapped = wrapped[: lines + 2]
    if len(parts) >= lines and wrapped:
        wrapped[-1] = wrapped[-1] + " …"
    return "\n".join(wrapped)


def draw_prediction_badges(ax: plt.Axes, case: dict[str, Any]) -> None:
    labels = [
        ("Few-shot", case["Few-shot"]),
        ("Neural", case["ProvMind-Neural-Only"]),
        ("Symbolic", case["ProvMind-Symbolic-Only"]),
        ("Hybrid", case["ProvMind-Hybrid"]),
    ]
    gt = str(case["gt"]).strip().upper()
    x0 = 0.62
    y0 = 0.82
    for idx, (label, pred) in enumerate(labels):
        correct = str(pred).strip().upper() == gt
        edge = COLORS["success"] if correct else COLORS["failure"]
        face = "#ECFDF3" if correct else "#FEF3F2"
        y = y0 - idx * 0.11
        patch = FancyBboxPatch(
            (x0, y),
            0.23,
            0.08,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            transform=ax.transAxes,
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x0 + 0.012, y + 0.04, f"{label}: {pred}", transform=ax.transAxes, va="center", ha="left", fontsize=9.5, fontweight="bold", color=edge)


def draw_score_table(ax: plt.Axes, scores: dict[str, Any], gt: str) -> None:
    letters = ["A", "B", "C", "D"]
    vals = np.array([float(scores.get(letter, 0.0)) for letter in letters], dtype=float)
    max_val = max(vals.max(), 1.0)
    ax.text(0.62, 0.41, "Symbolic scores", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=COLORS["text"])
    for idx, (letter, val) in enumerate(zip(letters, vals)):
        y = 0.36 - idx * 0.055
        ax.text(0.62, y, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=COLORS["text"])
        ax.add_patch(
            FancyBboxPatch(
                (0.665, y - 0.014),
                0.18 * (val / max_val),
                0.028,
                boxstyle="round,pad=0.0,rounding_size=0.01",
                transform=ax.transAxes,
                linewidth=0,
                facecolor=COLORS["hybrid"] if letter == gt else COLORS["baseline_mid"],
            )
        )
        ax.text(0.86, y, f"{val:.0f}", transform=ax.transAxes, fontsize=9, ha="right", color=COLORS["muted"])


def draw_case(ax: plt.Axes, case: dict[str, Any], panel_label: str) -> None:
    panel_bg(ax, rounded=True)
    border = COLORS["success"] if case["success"] else COLORS["failure"]
    for spine in ax.spines.values():
        spine.set_visible(False)
    rect = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        transform=ax.transAxes,
        linewidth=1.6,
        edgecolor=border,
        facecolor=COLORS["panel_bg"],
        zorder=-10,
        clip_on=False,
    )
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    title = f"{panel_label}  {case['task']} {'Success' if case['success'] else 'Failure'}"
    ax.text(0.02, 0.98, title, transform=ax.transAxes, va="top", ha="left", fontsize=15, fontweight="bold", color=COLORS["text"])
    ax.plot([0.02, 0.98], [0.92, 0.92], transform=ax.transAxes, color=COLORS["grid"], lw=1.0)

    ax.text(0.02, 0.90, shorten_question(case["question"]), transform=ax.transAxes, va="top", ha="left", fontsize=9.4, color=COLORS["text"])
    ax.text(0.02, 0.58, "Choices", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=COLORS["text"])
    gt = str(case["gt"]).strip().upper()
    for idx, choice in enumerate(case["choices"]):
        letter = chr(65 + idx)
        color = COLORS["success"] if letter == gt else COLORS["muted"]
        text = truncate_wrapped(f"{letter}. {choice}", width=38, max_lines=2)
        ax.text(0.02, 0.53 - idx * 0.075, text, transform=ax.transAxes, fontsize=8.8, color=color)

    ax.text(0.62, 0.90, "Predictions", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=COLORS["text"])
    draw_prediction_badges(ax, case)
    draw_score_table(ax, case["symbolic_scores"], gt)
    ax.text(0.62, 0.14, "Retrieved process analogs", transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=COLORS["text"])
    retrieved = case["retrieved_processes"] or ["N/A"]
    for idx, doi in enumerate(retrieved[:2]):
        ax.text(0.62, 0.09 - idx * 0.04, f"• {doi}", transform=ax.transAxes, fontsize=9.2, color=COLORS["muted"])
    if case["visible_edges"]:
        edge = " -> ".join(map(str, case["visible_edges"][0]))
        ax.text(0.02, 0.02, truncate_wrapped(f"Visible edge: {edge}", width=64, max_lines=1), transform=ax.transAxes, fontsize=8.2, color=COLORS["muted"], family="monospace")


def draw_fig5(cases: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 2, left=0.035, right=0.98, bottom=0.05, top=0.97, wspace=0.08, hspace=0.08)
    axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[1, 0]),
        "D": fig.add_subplot(gs[1, 1]),
    }
    for panel in ["A", "B", "C", "D"]:
        draw_case(axes[panel], cases.loc[panel].to_dict(), panel)
    save_figure(fig, "fig5_qualitative_cases")


def main() -> None:
    same_rows = collect_same_split_metrics()
    ood_rows = collect_cross_split_metrics()
    symbolic_rows = collect_symbolic_ablation_metrics()
    view_rows = collect_view_ablation_metrics()
    task_rows = collect_task_metrics()
    case_rows = collect_case_records()

    all_rows = same_rows + ood_rows + symbolic_rows + view_rows + task_rows
    metrics_df = pd.DataFrame(all_rows)
    cases_df = pd.DataFrame(case_rows).set_index("panel")

    export_jsonl(OUTPUT_DIR / "normalized_metrics.jsonl", all_rows)
    export_jsonl(OUTPUT_DIR / "normalized_cases.jsonl", case_rows)
    export_captions()

    draw_fig1(metrics_df)
    draw_fig2(metrics_df)
    draw_fig3(metrics_df)
    draw_fig4(metrics_df)
    draw_fig5(cases_df)

    print(f"Wrote figures and metadata to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
