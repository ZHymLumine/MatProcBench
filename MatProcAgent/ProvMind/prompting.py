from __future__ import annotations

from typing import Any


class ProcessReasoningPromptBuilder:
    """Reusable prompt builder for train-memory process reasoning agents."""

    def __init__(self, agent_name: str = "TraceFlow") -> None:
        self.agent_name = agent_name

    def build_plan_messages(
        self,
        task: str,
        record: dict[str, Any],
        retrieved_context: str,
        symbolic: dict[str, Any],
    ) -> list[dict[str, str]]:
        content = (
            f"Task type: {task}\n"
            f"Question:\n{record['question']}\n\n"
            f"Retrieved train-process context:\n{retrieved_context}\n\n"
            f"Symbolic task hint: {symbolic.get('task_hint', '')}\n"
            f"Symbolic option scores: {symbolic.get('option_scores', {})}\n\n"
            "Produce a concise 2-4 step reasoning plan. Do not answer the question yet."
        )
        return [
            {
                "role": "system",
                "content": (
                    f"You are {self.agent_name}, a materials process reasoning agent. "
                    "Plan how to use retrieved train-process analogs and symbolic hints "
                    "to answer a multiple-choice synthesis question."
                ),
            },
            {"role": "user", "content": content},
        ]

    def build_answer_messages(
        self,
        task: str,
        record: dict[str, Any],
        retrieved_context: str,
        symbolic: dict[str, Any],
        plan_text: str,
    ) -> list[dict[str, str]]:
        options_block = "\n".join(f"{letter}: {text}" for letter, text in record["choices"].items())
        score_lines = "\n".join(
            f"{letter}: {score}" for letter, score in sorted(symbolic.get("option_scores", {}).items())
        )
        content = (
            f"Task type: {task}\n"
            f"Question:\n{record['question']}\n\n"
            f"Options:\n{options_block}\n\n"
            f"Retrieved train-process context:\n{retrieved_context}\n\n"
            f"Symbolic task hint: {symbolic.get('task_hint', '')}\n"
            f"Symbolic option scores:\n{score_lines}\n\n"
            f"Reasoning plan:\n{plan_text}\n\n"
            "Choose the best option using the retrieved train-process evidence and symbolic scores. "
            "Return only one line in the format 'Answer: X' where X is A, B, C, or D."
        )
        return [
            {
                "role": "system",
                "content": (
                    f"You are {self.agent_name}, a fair training-free materials-process agent using train-split "
                    "process memory. You must answer multiple-choice questions strictly from the provided "
                    "retrieved train-process context and symbolic evidence. If evidence is weak, prefer the "
                    "highest-scoring symbolic option."
                ),
            },
            {"role": "user", "content": content},
        ]

    def build_decision_messages(
        self,
        task: str,
        record: dict[str, Any],
        retrieved_context: str = "",
        symbolic: dict[str, Any] | None = None,
        plan_text: str = "",
    ) -> list[dict[str, str]]:
        options_block = "\n".join(f"{letter}: {text}" for letter, text in record["choices"].items())

        sections = [
            f"Task type: {task}",
            f"Question:\n{record['question']}",
            f"Options:\n{options_block}",
        ]

        if retrieved_context.strip():
            sections.append(f"Retrieved train-process context:\n{retrieved_context}")

        if symbolic:
            task_hint = symbolic.get("task_hint", "")
            score_lines = "\n".join(
                f"{letter}: {score}" for letter, score in sorted(symbolic.get("option_scores", {}).items())
            )
            if task_hint:
                sections.append(f"Symbolic task hint: {task_hint}")
            if score_lines:
                sections.append(f"Symbolic option scores:\n{score_lines}")

        if plan_text.strip():
            sections.append(f"Reasoning plan:\n{plan_text}")

        sections.append(
            "Choose the best option and return only one line in the format 'Answer: X' "
            "where X is A, B, C, or D."
        )

        content = "\n\n".join(sections)
        return [
            {
                "role": "system",
                "content": (
                    f"You are {self.agent_name}, a materials-process reasoning system. "
                    "Use only the provided evidence, if any, and answer the multiple-choice question. "
                    "If no external evidence is provided, rely on the question alone."
                ),
            },
            {"role": "user", "content": content},
        ]
