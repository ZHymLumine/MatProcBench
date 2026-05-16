from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline

from .compiler import ProcessState
from .memory import ProcessMemoryIndex, TrainProcessIndex
from .prompting import ProcessReasoningPromptBuilder
from .scoring import TaskSignalScorer
from .utils import extract_answer_letter, infer_task, load_jsonl, render_sequence


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class TrainOnlyProcessMemoryAgent:
    """Generic train-memory process-reasoning agent with symbolic priors."""

    METHOD_NAME = "traceflow"

    def __init__(
        self,
        index: ProcessMemoryIndex,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 8,
        plan_max_new_tokens: int = 96,
        answer_max_new_tokens: int = 48,
        load_in_4bit: bool = False,
        use_retrieval: bool = True,
        use_symbolic: bool = True,
        use_planning: bool = True,
        use_symbolic_fallback: bool = True,
        scorer: TaskSignalScorer | None = None,
        prompt_builder: ProcessReasoningPromptBuilder | None = None,
    ) -> None:
        self.index = index
        self.model_name = model_name
        self.top_k = top_k
        self.plan_max_new_tokens = plan_max_new_tokens
        self.answer_max_new_tokens = answer_max_new_tokens
        self.use_retrieval = use_retrieval
        self.use_symbolic = use_symbolic
        self.use_planning = use_planning
        self.use_symbolic_fallback = use_symbolic_fallback
        self.scorer = scorer or TaskSignalScorer(index)
        self.prompt_builder = prompt_builder or ProcessReasoningPromptBuilder(agent_name="TraceFlow")
        self.pipe = self._init_llm(model_name, load_in_4bit=load_in_4bit)

    @classmethod
    def from_files(
        cls,
        raw_file: str,
        train_file: str,
        model_name: str = DEFAULT_MODEL,
        top_k: int = 8,
        plan_max_new_tokens: int = 96,
        answer_max_new_tokens: int = 48,
        load_in_4bit: bool = False,
        use_retrieval: bool = True,
        use_symbolic: bool = True,
        use_planning: bool = True,
        use_symbolic_fallback: bool = True,
    ) -> "TrainOnlyProcessMemoryAgent":
        index = ProcessMemoryIndex.from_files(raw_file, train_file)
        return cls(
            index=index,
            model_name=model_name,
            top_k=top_k,
            plan_max_new_tokens=plan_max_new_tokens,
            answer_max_new_tokens=answer_max_new_tokens,
            load_in_4bit=load_in_4bit,
            use_retrieval=use_retrieval,
            use_symbolic=use_symbolic,
            use_planning=use_planning,
            use_symbolic_fallback=use_symbolic_fallback,
        )

    def _init_llm(self, model_name: str, load_in_4bit: bool) -> Any:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        quantization_config = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            if load_in_4bit else None
        )
        return pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=None if load_in_4bit else torch.bfloat16,
            model_kwargs={"quantization_config": quantization_config} if quantization_config else {},
            device_map="auto",
            trust_remote_code=True,
        )

    def _run_llm(self, messages: list[dict[str, str]], max_new_tokens: int) -> str:
        outputs = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            num_return_sequences=1,
        )
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            return generated[-1]["content"].strip()
        return str(generated).strip()

    def _best_symbolic_letter(self, scores: dict[str, float]) -> str:
        return max(scores.items(), key=lambda item: (item[1], -ord(item[0])))[0]

    def _empty_symbolic(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "task_hint": "",
            "option_scores": {
                letter: 0.0 for letter in record.get("choices", {})
            },
        }

    def _format_retrieved_context(self, retrieved: list[ProcessState]) -> str:
        blocks: list[str] = []
        for idx, process in enumerate(retrieved[:5], start=1):
            precursors = ", ".join(entity.label for _eid, entity in process.precursors() if entity.label)
            route = render_sequence(process.route_labels(instance_aware=False))
            blocks.append(
                f"[Retrieved Train Process {idx}]\n"
                f"DOI: {process.doi}\n"
                f"Process: {process.label}\n"
                f"Precursors: {precursors or 'unknown'}\n"
                f"Route: {route}\n"
                f"Steps: {process.step_count()}"
            )
        return "\n\n".join(blocks) if blocks else "No retrieved train processes."

    def answer_record(self, record: dict[str, Any]) -> dict[str, Any]:
        task = infer_task(record)
        retrieved = (
            self.index.retrieve_processes(record, task, top_k=self.top_k)
            if self.use_retrieval and self.top_k > 0
            else []
        )
        symbolic = self.scorer.score(task, record, retrieved) if self.use_symbolic else self._empty_symbolic(record)
        retrieved_context = self._format_retrieved_context(retrieved) if self.use_retrieval else ""
        symbolic_best = self._best_symbolic_letter(symbolic["option_scores"])

        plan_text = ""
        if self.use_planning:
            plan_messages = self.prompt_builder.build_plan_messages(
                task,
                record,
                retrieved_context,
                symbolic if self.use_symbolic else {},
            )
            plan_text = self._run_llm(plan_messages, max_new_tokens=self.plan_max_new_tokens)

        answer_messages = self.prompt_builder.build_decision_messages(
            task,
            record,
            retrieved_context=retrieved_context,
            symbolic=symbolic if self.use_symbolic else None,
            plan_text=plan_text,
        )
        llm_raw_answer = self._run_llm(answer_messages, max_new_tokens=self.answer_max_new_tokens)
        llm_letter = extract_answer_letter(llm_raw_answer)
        used_symbolic_fallback = False
        if llm_letter in {"A", "B", "C", "D"}:
            final_answer = llm_letter
        elif self.use_symbolic and self.use_symbolic_fallback:
            final_answer = symbolic_best
            used_symbolic_fallback = True
        else:
            final_answer = llm_raw_answer

        return {
            "qid": record.get("qid", ""),
            "task": record.get("task", ""),
            "method": self.METHOD_NAME,
            "question": record["question"],
            "model_answer": final_answer,
            "gt_answer": record.get("answer", ""),
            "choices": [record["choices"][label] for label in "ABCD" if label in record["choices"]],
            "answer_index": ord(record.get("answer", " ")) - ord("A") if record.get("answer", "") in "ABCD" else -1,
            "traceflow_trace": {
                "agent": "TraceFlow",
                "setting": "fair_train_process_only",
                "model": self.model_name,
                "task": task,
                "retrieved_processes": [{"doi": proc.doi, "process_label": proc.label} for proc in retrieved[:5]],
                "symbolic": symbolic,
                "symbolic_best": symbolic_best,
                "config": {
                    "top_k": self.top_k,
                    "use_retrieval": self.use_retrieval,
                    "use_symbolic": self.use_symbolic,
                    "use_planning": self.use_planning,
                    "use_symbolic_fallback": self.use_symbolic_fallback,
                },
                "llm_plan": plan_text,
                "llm_raw_answer": llm_raw_answer,
                "used_symbolic_fallback": used_symbolic_fallback,
            },
        }


class TraceFlowAgent(TrainOnlyProcessMemoryAgent):
    """Backward-compatible public wrapper for the TraceFlow method."""


_infer_task = infer_task
