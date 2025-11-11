# agents/bdd_agent.py

from langchain_openai import AzureChatOpenAI
import os
from Agents.llm import LLM
from dotenv import load_dotenv, find_dotenv
load_dotenv()

class BDDDecomposerAgent:
    def __init__(self):
        self.LLM = LLM

    def decompose(self, task_description: str) -> str:
        prompt = f"""
                    You are a software engineer skilled in Behavior Driven Development (BDD). 
                    Decompose the following feature into Given-When-Then format:

                    "{task_description}"

                    Use bullet points for each scenario.
                    """
        messages=[
            {"role": "system", "content": "You write BDD test scenarios."},
            {"role": "user", "content": prompt}
        ],
        response=self.LLM.invoke(
            f'''{messages}'''
        ).content
        
        return response
