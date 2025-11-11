import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO
import fitz  # PyMuPDF
import torch
import transformers
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from googletrans import Translator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
import os

nltk.download('punkt')
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('agg')

# ---------------------------
# Setup summarization model
# ---------------------------
@st.cache_resource
def load_summarization_pipeline():
    summarization_pipeline = transformers.pipeline(
        model="facebook/bart-large-cnn",
        tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large-cnn"),
        task="summarization",
        max_length=250,
    )
    return HuggingFacePipeline(pipeline=summarization_pipeline)

LLM = load_summarization_pipeline()

# ---------------------------
# Utility Functions
# ---------------------------

def summarize_text(text):
    data = "Summarize This Data in 2 to 5 sentences: " + text
    summary = LLM.invoke(data)
    return summary

def translate_text(text, target_language='en'):
    translator = Translator()
    translated = translator.translate(text, dest=target_language)
    return translated.text

def extract_target_language(text):
    tokens = text.lower().split()
    if "translate" in tokens and "to" in tokens:
        idx = tokens.index("to") + 1
        if idx < len(tokens):
            return tokens[idx]
    return None

def generate_summary_from_pdf(pdf_content, sentences_count=5):
    document = fitz.open(stream=pdf_content, filetype="pdf")
    extracted_text = ""
    for page_number in range(len(document)):
        page = document.load_page(page_number)
        extracted_text += page.get_text()
    document.close()
    summary = generate_summary(extracted_text, sentences_count)
    return summary

def generate_summary(text, sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    summary_text = " ".join(str(sentence) for sentence in summary)
    return summary_text

def automated_data_analysis(df):
    st.subheader("Automated Data Analysis")
    df = df.head(1500)
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Histogram
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            plt.title(f'Histogram of {column}')
            st.pyplot(fig)

            # Boxplot
            fig, ax = plt.subplots()
            sns.boxplot(x=df[column], ax=ax)
            plt.title(f'Boxplot of {column}')
            st.pyplot(fig)

        elif pd.api.types.is_object_dtype(df[column]):
            st.write(f"**Value counts for {column}:**")
            st.write(df[column].value_counts())

def LLM_agent(input_text):
    if any(keyword in input_text.lower() for keyword in ['translate', 'translation']):
        target_language = extract_target_language(input_text)
        if target_language:
            clean_text = input_text.lower().replace("translate", "").replace("to", "").replace(target_language, "")
            translated_text = translate_text(clean_text, target_language)
            return f"**Translated text to {target_language}:**\n\n{translated_text}"
        else:
            return "⚠️ Please specify the target language like: 'translate to French'"
    else:
        summary = summarize_text(input_text)
        return f"**Summary:**\n\n{summary}"

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="LLM Agent Orchestrator", layout="wide")
st.title("🤖 LLM Agent – Dynamic Task Orchestrator")

st.write("""
Upload a **PDF** for summarization, a **CSV** for automated data analysis, or 
enter **text** for translation/summarization using LLMs.
""")

uploaded_file = st.file_uploader("Upload a file (PDF or CSV)", type=["pdf", "csv"])
input_text = st.text_area("Or enter text input", height=150)
run_button = st.button("Run Agent")

if run_button:
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            with st.spinner("Reading and summarizing PDF..."):
                pdf_content = uploaded_file.read()
                output = generate_summary_from_pdf(pdf_content, sentences_count=5)
                st.success("✅ PDF Summarization Complete")
                st.write(output)

        elif uploaded_file.name.endswith('.csv'):
            with st.spinner("Performing data analysis..."):
                df = pd.read_csv(uploaded_file)
                automated_data_analysis(df)
    elif input_text.strip():
        with st.spinner("Processing text with LLM Agent..."):
            output = LLM_agent(input_text)
            st.markdown(output)
    else:
        st.warning("Please upload a file or enter some text to proceed.")
