from main import LLMOrchestrator
from dotenv import load_dotenv, find_dotenv
load_dotenv()


orchestrator = LLMOrchestrator()



# # Example input for translation
# input_text = "Translate the text 'How are you?' to Portuguese"
# print(orchestrator.route(input_text))
# # Expected OUTPUT: Como vai

# # Example input for summarization
# long_text = "Artificial intelligence is transforming industries. It enables automation, decision making..."
# print(orchestrator.route(long_text))

# # Example input for BDD decomposition
# bdd_input = "As a user, I want to reset my password so I can regain access to my account."
# print(orchestrator.route(bdd_input))

# Example input for EDA (must specify file_path)
path=r'C:\Users\z0052mvs\Desktop\New Microsoft Excel Worksheet.xlsx'
try:
    eda_input = f"Analyze the Excel file at path {path}"
    print(orchestrator.route(eda_input))
except Exception as e:
    print(f"The error: {e}")
