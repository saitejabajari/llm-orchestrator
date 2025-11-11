# agents/translate_agent.py

import requests
from dotenv import load_dotenv, find_dotenv
load_dotenv()
# class TranslateAgent:
#     def __init__(self, api_key: str):
#         self.api_key = api_key
#         self.endpoint = "https://translation.googleapis.com/language/translate/v2"

#     def translate(self, text: str, target_language: str = "en") -> str:
#         params = {
#             'q': text,
#             'target': target_language,
#             'format': 'text',
#             'key': self.api_key
#         }
#         response = requests.post(self.endpoint, data=params)
#         result = response.json()
#         try:
#             return result['data']['translations'][0]['translatedText']
#         except (KeyError, IndexError):
#             return f"Translation failed: {result}"

import requests

class TranslateAgent:

    def translate(self,text, target_language, source_language='auto'):

        url = f"https://lingva.ml/api/v1/{source_language}/{target_language}/{text}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()['translation']
        else:
            return f"Translation failed: {response.status_code}, {response.text}"


# p=TranslateAgent()
# # Example
# print(p.translate("Hola, ¿cómo estás?",target_language="pt"))


