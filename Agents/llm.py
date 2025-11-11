
from langchain_ollama import ChatOllama

LLM = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    # other params...
)
# Example usage:
response = LLM.invoke("Hello")
print(response.content)

# print(LLM.invoke("Hello").content)