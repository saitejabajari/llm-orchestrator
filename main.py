# from Agents.bdd import BDDDecomposerAgent
# from Agents.da import EDAAgent
# from Agents.summarizer import SummarizerAgent
# from Agents.translate import TranslateAgent

# config={
#     "1":f"{BDDDecomposerAgent()}",
#     "2": f"{EDAAgent()}",
#     "3":f"{SummarizerAgent()}",
#     "4":f"{TranslateAgent()}"
# }

# orchestrator_llm.py

import json
from langchain_openai import AzureChatOpenAI
from Agents.bdd import BDDDecomposerAgent
from Agents.da import EDAAgent
from Agents.summarizer import SummarizerAgent
from Agents.translate import TranslateAgent
import os
from Agents.llm import LLM
from dotenv import load_dotenv, find_dotenv
load_dotenv()
class LLMOrchestrator:
    def __init__(self, google_translate_api_key=None):

        self.translate_agent = TranslateAgent()
        self.eda_agent = EDAAgent()
        self.summarizer_agent = SummarizerAgent()
        self.bdd_agent = BDDDecomposerAgent()

    def get_routing_instruction(self, user_input) -> dict:
        prompt = f"""
You are an AI orchestrator. Your job is to read user input and choose the correct tool and parameters.

Available tools:
- TranslateAgent: For language translation.
- EDAAgent: For analyzing Excel (.xlsx) files.
- SummarizerAgent: For summarizing long text.
- BDDDecomposerAgent: For breaking down software feature requests into BDD scenarios.

Respond ONLY in valid JSON with two keys:
- "agent": one of ["TranslateAgent", "EDAAgent", "SummarizerAgent", "BDDDecomposerAgent"]
- "params": an object containing the parameters required for the agent.

Examples:
Input: "translate the text 'Hello' to Portuguese"
Response:
{{
  "agent": "TranslateAgent",
  "params": {{
    "text": "Hello",
    "target_language": "pt"
  }}
}}

Input: "Summarize the following article ..."
Response:
{{
  "agent": "SummarizerAgent",
  "params": {{
    "text": "...article text..."
  }}
}}

Input:
\"\"\"{user_input}\"\"\"
"""

        messages=[
            {"role": "system", "content": "You are an intelligent routing agent."},
            {"role": "user", "content": prompt}
        ],
           
        response=LLM.invoke(f'''{messages}''')
        cleaned_response = response.content.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        return cleaned_response

    def route(self, input_data, **kwargs):
        response_text = self.get_routing_instruction(input_data)

        try:
            response_json = json.loads(response_text)
            agent_name = response_json.get("agent")
            params = response_json.get("params", {})
        except json.JSONDecodeError:
            return f"Error: Could not parse LLM response as JSON.\nResponse was:\n{response_text}"

        if agent_name == "TranslateAgent":
            text = params.get("text", "")
            target_language = params.get("target_language", "en")
            return self.translate_agent.translate(text, target_language)

        elif agent_name == "EDAAgent":
            file_path = params.get("file_path", "")
            return self.eda_agent.analyze(file_path)

        elif agent_name == "SummarizerAgent":
            text = params.get("text", "")
            return self.summarizer_agent.summarize(text)

        elif agent_name == "BDDDecomposerAgent":
            text = params.get("text", "")
            return self.bdd_agent.decompose(text)

        else:
            return f"Error: Unknown agent '{agent_name}' returned by LLM."
