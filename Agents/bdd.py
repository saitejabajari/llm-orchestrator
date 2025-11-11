# agents/bdd_agent.py

import os
from Agents.llm import LLM
from dotenv import load_dotenv

load_dotenv()


class BDDDecomposerAgent:
    """Simple wrapper around the project's LLM to produce BDD scenarios.

    This module previously imported `langchain_openai` which caused import
    errors in environments without that package. The agent uses the project's
    `Agents.llm.LLM` instance (which is Ollama-backed in this repo).
    """

    def __init__(self):
        self.LLM = LLM

    def decompose(self, task_description: str) -> str:
        prompt = f"""
You are a software engineer skilled in Behavior Driven Development (BDD).
Decompose the following feature into Given-When-Then format:

"{task_description}"

Provide one or more scenarios, using bullet points for each scenario.
"""

        messages = [
            {"role": "system", "content": "You write clear BDD test scenarios."},
            {"role": "user", "content": prompt},
        ]

        # Use the project's LLM wrapper. It expects a prompt-like string.
        response = self.LLM.invoke(f"{messages}")
        return getattr(response, "content", str(response))
