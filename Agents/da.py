# agents/eda_agent.py

import pandas as pd
from dotenv import load_dotenv, find_dotenv
load_dotenv()
class EDAAgent:
    def __init__(self):
        pass

    def analyze(self, file_path: str) -> dict:
        df = pd.read_excel(file_path)
        summary = {
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "describe": df.describe(include='all').to_dict()
        }
        return summary
