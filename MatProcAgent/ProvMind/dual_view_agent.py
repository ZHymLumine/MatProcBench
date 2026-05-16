"""
dual_view_agent.py — DualViewAgent for ProvMind.

Extends TrainOnlyProcessMemoryAgent by swapping in a DualViewIndex, which
replaces the heuristic retriever with a dual-view (text encoder + 2-layer GAT)
retriever.  The scoring step is configurable via scoring_mode:

  "symbolic"  — original TaskSignalScorer (string-matching + bigrams)
  "neural"    — NeuralTaskScorer (embedding NN over step/process embeddings)
  "hybrid"    — weighted blend of symbolic and neural (default)
"""

from __future__ import annotations

from typing import Any

from .agent import DEFAULT_MODEL, TrainOnlyProcessMemoryAgent
from .dual_view_retriever import DualViewIndex, RetrievalMode
from .neural_scoring import HybridTaskScorer, NeuralTaskScorer, ScoringMode
from .scoring import TaskSignalScorer
from .utils import load_jsonl


class DualViewAgent(TrainOnlyProcessMemoryAgent):
    """
    TraceFlow agent with dual-view retrieval and configurable neural scoring.

    retrieval_mode:
        "heuristic"  — original string-matching (same as TraceFlowAgent)
        "text_only"  — SentenceTransformer cosine similarity only
        "dual"       — text + 2-layer frozen GAT + heuristic (default)

    scoring_mode:
        "symbolic"   — original TaskSignalScorer
        "neural"     — NeuralTaskScorer (embedding NN)
        "hybrid"     — weighted blend of symbolic + neural (default)
    """

    METHOD_NAME = "dualview"

    @classmethod
    def from_files(  # type: ignore[override]
        cls,
        raw_file: str,
        train_file: str,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 8,
        retrieval_mode: RetrievalMode = "dual",
        text_encoder_name: str = "all-mpnet-base-v2",
        alpha: float = 0.4,
        beta: float = 0.3,
        gamma: float = 0.3,
        gat_hidden_dim: int = 256,
        gat_heads: int = 4,
        gat_out_dim: int = 256,
        cache_dir: str | None = None,
        plan_max_new_tokens: int = 96,
        answer_max_new_tokens: int = 48,
        load_in_4bit: bool = False,
        use_symbolic: bool = True,
        use_planning: bool = True,
        use_symbolic_fallback: bool = True,
        scoring_mode: ScoringMode = "hybrid",
        scoring_sym_weight: float = 0.5,
        scoring_neu_weight: float = 0.5,
    ) -> "DualViewAgent":
        print(f"[DualViewAgent] Building DualViewIndex (retrieval={retrieval_mode}) …")
        index = DualViewIndex.from_files(
            raw_file=raw_file,
            train_file=train_file,
            text_encoder_name=text_encoder_name,
            retrieval_mode=retrieval_mode,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            gat_hidden_dim=gat_hidden_dim,
            gat_heads=gat_heads,
            gat_out_dim=gat_out_dim,
            cache_dir=cache_dir,
        )

        # Build scorer according to scoring_mode
        scorer: TaskSignalScorer | HybridTaskScorer
        if scoring_mode == "symbolic" or retrieval_mode == "heuristic":
            scorer = TaskSignalScorer(index)
            print(f"[DualViewAgent] Scoring mode: symbolic")
        elif scoring_mode == "neural":
            scorer = NeuralTaskScorer(index)
            print(f"[DualViewAgent] Scoring mode: neural")
        else:  # hybrid
            sym = TaskSignalScorer(index)
            neu = NeuralTaskScorer(index)
            scorer = HybridTaskScorer(sym, neu, scoring_sym_weight, scoring_neu_weight)
            print(f"[DualViewAgent] Scoring mode: hybrid "
                  f"(sym={scoring_sym_weight}, neu={scoring_neu_weight})")

        return cls(
            index=index,
            model_name=model_name,
            top_k=top_k,
            plan_max_new_tokens=plan_max_new_tokens,
            answer_max_new_tokens=answer_max_new_tokens,
            load_in_4bit=load_in_4bit,
            use_retrieval=True,
            use_symbolic=use_symbolic,
            use_planning=use_planning,
            use_symbolic_fallback=use_symbolic_fallback,
            scorer=scorer,
        )

    def answer_record(self, record: dict[str, Any]) -> dict[str, Any]:
        result = super().answer_record(record)
        trace = result["traceflow_trace"]
        trace["agent"]          = "DualView"
        trace["retrieval_mode"] = self.index.retrieval_mode          # type: ignore[attr-defined]
        trace["retrieval_fusion_weights"] = {
            "alpha": self.index._alpha,                              # type: ignore[attr-defined]
            "beta":  self.index._beta,                               # type: ignore[attr-defined]
            "gamma": self.index._gamma,                              # type: ignore[attr-defined]
        }
        scorer = self.scorer
        if isinstance(scorer, HybridTaskScorer):
            trace["scoring_mode"] = "hybrid"
            trace["scoring_weights"] = {"sym": scorer._sym_w, "neu": scorer._neu_w}
        elif isinstance(scorer, NeuralTaskScorer):
            trace["scoring_mode"] = "neural"
        else:
            trace["scoring_mode"] = "symbolic"
        return result
