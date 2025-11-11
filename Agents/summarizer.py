# agents/summarizer_agent.py


from langchain_openai import AzureChatOpenAI
import os
from Agents.llm import LLM
from dotenv import load_dotenv, find_dotenv
load_dotenv()
class SummarizerAgent:
    def __init__(self):
        # self.LLM : AzureChatOpenAI =None
        # self.LLM = AzureChatOpenAI(
        #     azure_deployment=os.getenv("AZURE_DEPLOYMENT"),
        #     openai_api_version=os.getenv("OPENAI_API_VERSION"),
        #     api_key=os.getenv("AZURE_OPENAI_KEY"),
        #     azure_endpoint=os.getenv('AZURE_ENDPOINT'),
        # )
        self.LLM=LLM

    def summarize(self, text: str) -> str:
        messages=[
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
        ],
        response=self.LLM.invoke(
            f'''{messages}'''
        ).content
        
        return response
