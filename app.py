import matplotlib
from flask import Flask, render_template, request
from io import StringIO
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import transformers
from googletrans import Translator
from transformers import pipeline, AutoTokenizer
from langchain import HuggingFacePipeline
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fitz  # PyMuPDF
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from werkzeug.utils import secure_filename
import nltk
nltk.download('punkt')
matplotlib.use('agg')


app = Flask(__name__)


summarization_pipeline = transformers.pipeline(
    model="facebook/bart-large-cnn",
    tokenizer=AutoTokenizer.from_pretrained("facebook/bart-large-cnn"),
    task="summarization",
    max_length=250,
)


def summarize(text):
    LLM = HuggingFacePipeline(pipeline=summarization_pipeline)
    data = "Summarize This Data in 2 to 5 sentences : " + text
    summary = LLM.invoke(data)
    return summary


def translate(text, target_language='en'):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language)
    return translated_text.text


def extract_target_language(text):
    tokens = text.lower().split()
    if "translate" in tokens and "to" in tokens:
        target_language_index = tokens.index("to") + 1
        target_language = tokens[target_language_index]
        return target_language
    else:
        return None


def LLM_agent(input_text):

    if any(keyword in input_text.lower() for keyword in ['translate', 'translation']):
        target_language = extract_target_language(input_text)
        if target_language:
            input_text = input_text.lower().replace("translate", "")
            input_text = input_text.lower().replace("to", "")
            input_text = input_text.lower().replace(target_language, "")
            translated_text = translate(input_text, target_language)
            return f"Translated text to {target_language}: {translated_text}"
        else:
            return "Target language not specified. Please provide the target language in the format 'translate to <language>'."
    else:
        summary = summarize(input_text)
        return f"Summary: {summary}"


# def automated_data_analysis(df):
#     # Display the first few rows of the dataset
#     print("First few rows of the dataset:")
#     print(df.head())

#     # Data visualization
#     for column in df.columns:
#         if pd.api.types.is_numeric_dtype(df[column]):
#             # Histogram
#             plt.figure()
#             sns.histplot(df[column])
#             plt.title(f'Histogram of {column}')
#             plt.show()

#             # Boxplot
#             plt.figure()
#             sns.boxplot(x=df[column])
#             plt.title(f'Boxplot of {column}')
#             plt.show()
#         elif pd.api.types.is_object_dtype(df[column]):
#             # Value counts for categorical columns
#             print(f'\nValue counts for {column}:')
#             print(df[column].value_counts())

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
    summary_text = ""
    for sentence in summary:
        summary_text += str(sentence) + " "
    return summary_text


def automated_data_analysis(df):
    analysis_results = []
    df = df.head(1500)
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Histogram
            plt.figure()
            sns.histplot(df[column])
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            # Save the histogram plot
            hist_plot_path = f'static/hist_plot_{column}.png'
            plt.savefig(hist_plot_path)
            plt.close()  # Close the plot to release memory

            # Boxplot
            plt.figure()
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)
            # Save the boxplot
            box_plot_path = f'static/box_plot_{column}.png'
            plt.savefig(box_plot_path)
            plt.close()  # Close the plot to release memory

            analysis_results.append(
                {'column': column, 'hist_plot': hist_plot_path, 'box_plot': box_plot_path})
        elif pd.api.types.is_object_dtype(df[column]):
            # Value counts for categorical columns
            value_counts = df[column].value_counts()
            analysis_results.append(
                {'column': column, 'value_counts': value_counts})

    return analysis_results


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result', methods=["POST"])
def AgentLLM():
    if 'inputfile_' in request.files and request.files['inputfile_'].filename:
        # Handling PDF file upload
        inputfile = request.files['inputfile_']
        # Check if the uploaded file is a PDF
        if inputfile.filename.endswith('.pdf'):
            file_content = inputfile.read()  # Read the content of the uploaded file
            output = generate_summary_from_pdf(file_content, sentences_count=5)
            return render_template("output.html", data={"Output": output})
        elif inputfile.filename.endswith('.csv'):
            # Read the content of the uploaded file
            file_content = inputfile.read().decode("utf-8")
            df = pd.read_csv(StringIO(file_content))
            analysis_results = automated_data_analysis(df)
            return render_template("analysis_results.html", analysis_results=analysis_results)
        else:
            return render_template("output.html", data={"Output": "Please upload a PDF file."})
    else:
        # Handling text input
        inputtext = request.form["inputtext_"]
        output = LLM_agent(inputtext)
        return render_template("output.html", data={"Output": output})


if __name__ == '__main__':
    app.run()
