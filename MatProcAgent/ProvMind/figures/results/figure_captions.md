# Figure Captions

## FIG1
Figure 1 | Same-split performance comparison on MatProcBench with Qwen2.5-7B-Instruct. Accuracy of Zero-shot, Few-shot, RAG, GraphRAG, SFT Zero-shot, SFT Few-shot, and ProvMind-Hybrid across the Year+Type, Random, Material Type, and Publication Year splits. ProvMind-Hybrid remains competitive across all same-split settings and yields the clearest improvement on the Random split while preserving consistent gains on the more challenging Material Type and Publication Year splits.

## FIG2
Figure 2 | Clean OOD generalization from Year+Type training. (A) Overall accuracy on the matched Year+Type split and the clean OOD Material Type and Publication Year splits. (B) Accuracy retention relative to the Year+Type test set. (C) Absolute gain of ProvMind-Hybrid over SFT Zero-shot on each clean evaluation split. By excluding the contamination-prone Random split, this figure isolates generalization under material-type and temporal distribution shift.

## FIG3
Figure 3 | Mechanistic ablations of ProvMind. (A) Ablations within the symbolic-only variant, isolating the contribution of planning, retrieval, symbolic scoring, and structured LLM-mediated decision making. (B) Task-family breakdown of the same symbolic-only variants, revealing which reasoning skills are most sensitive to removing each component. (C) View ablation of ProvMind, comparing neural-only, symbolic-only, and hybrid variants on the Additional, Type, and Year splits. These panels separate component-level contributions within symbolic reasoning from higher-level differences between neural, symbolic, and hybrid formulations of ProvMind.

## FIG4
Figure 4 | Task hierarchy and distribution-shift sensitivity of ProvMind-Hybrid. (A) Per-task accuracy across the Year+Type, Material Type, Publication Year, and Random splits. (B) Accuracy gap between the Random split and the average of the other three splits, highlighting tasks that are most sensitive to distribution shift. (C) Overall task ranking by non-random average accuracy. Causal ordering remains consistently easy, whereas route retrieval and full condition-set prediction remain the dominant bottlenecks under shift.

## FIG5
Figure 5 | Qualitative comparison on representative Type-split cases. Three successful examples show how provenance-aware scoring and retrieved process analogs correct mistakes made by standard prompting baselines, while one failure case illustrates a remaining weakness in long-route discrimination. For each example, we display a compressed question context, model predictions, and a minimal evidence view derived from symbolic scores and retrieved process traces.
