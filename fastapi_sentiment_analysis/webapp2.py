from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from pydantic import BaseModel
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pathlib
import textwrap
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import pandas as pd
import re  # Import regular expression module for hyperlink removal
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import PIL.Image
from wordcloud import WordCloud
import collections
import json
import torch
from fpdf import FPDF
from bertopic import BERTopic
import kaleido
import nest_asyncio
import re
import shutil
import os
from fpdf import FPDF
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredCSVLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware

if os.getenv("FASTAPI_ENV") == "development":
    nest_asyncio.apply()

question_responses = []
document_analyzed = False

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GetColumn(BaseModel):
    meta: dict
    columns: str

class AnalyzeDocumentRequest(BaseModel):
    api_key: str
    target_variable: str
    custom_stopwords: str
    custom_question: str

class AnalyzeDocumentResponse(BaseModel):
    meta: dict
    sentiment_plot_path: str
    topic_plot_path: str
    topic_plot_path1: str
    topic_plot_path2: str
    wordcloud_positive: str
    gemini_response_pos: str
    wordcloud_neutral: str
    gemini_response_neu: str
    wordcloud_negative: str
    gemini_response_neg: str
    bigram_positive: str
    gemini_response_pos1: str
    bigram_neutral: str
    gemini_response_neu1: str
    bigram_negative: str
    gemini_response_neg1: str
    unigram_positive: str
    gemini_response_pos2: str
    unigram_neutral: str
    gemini_response_neu2: str
    unigram_negative: str
    gemini_response_neg2: str
    pdf_file_path: str


class AskRequest(BaseModel):
    question: str
    api_key: str

class AskResponse(BaseModel):
    meta: dict
    question: str
    result: str

# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Create stopword remover
stop_factory = StopWordRemoverFactory()
more_stopword = ['yg', 'yang', 'aku', 'gw', 'gua', 'gue']
data = stop_factory.get_stop_words() + more_stopword

# Define hyperlink pattern for removal
hyperlink_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
number_pattern = re.compile(r'\b\d+\b')

emoticon_pattern = re.compile(u'('
    u'\ud83c[\udf00-\udfff]|'
    u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
    u'[\u2600-\u26FF\u2700-\u27BF])+', 
    re.UNICODE)

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/process", response_model=GetColumn)
async def process_file(request: Request, file: UploadFile = File(...)):
    global df
    file_location = f"static/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load DataFrame based on file type
    file_extension = os.path.splitext(file.filename)[1]
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_location, delimiter=",")
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_location)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")

        # Get columns of the DataFrame
        columns = df.columns.tolist()

        return GetColumn(
            meta={"status": "success", "code": 200},
            columns=", ".join(columns)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/analyze", response_model=AnalyzeDocumentResponse)
async def analyze(
    request: Request,
    api_key: str = Form(...),
    target_variable: str = Form(...),
    custom_stopwords: str = Form(""),
    file: UploadFile = File(...),
    custom_question: str = Form("")
):
    global df
    # Read the uploaded CSV file

    file_location = f"static/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load DataFrame based on file type
    file_extension = os.path.splitext(file.filename)[1]
    try:
        if file_extension == '.csv':
            df = pd.read_csv(file_location, delimiter=",")
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_location)
        else:
            raise HTTPException(status_code=415, detail="Unsupported file format")


        if target_variable not in df.columns:
            return "Selected target variable does not exist in the dataset."

        add_stopwords = ['the', 'of', 'is', 'a', 'in']
        custom_stopword_list = [word.strip() for word in custom_stopwords.split(',')]
        all_stopwords = data + custom_stopword_list + add_stopwords

        # Remove hyperlinks, emoticons, numbers, and stopwords
        hyperlink_pattern = r'https?://\S+|www\.\S+'
        emoticon_pattern = r'[:;=X][oO\-]?[D\)\]\(\]/\\OpP]'
        number_pattern = r'\b\d+\b'

        df[target_variable] = df[target_variable].astype(str)
        df['cleaned_text'] = df[target_variable].str.replace(hyperlink_pattern, '', regex=True)
        df['cleaned_text'] = df['cleaned_text'].str.replace(emoticon_pattern, '', regex=True)
        df['cleaned_text'] = df['cleaned_text'].str.replace(number_pattern, '', regex=True)
        for stopword in all_stopwords:
            df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() != stopword]))

        # Perform stopwords removal
        df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(
            [stemmer.stem(word) for word in stop_factory.create_stop_word_remover().remove(x).split()
            if word.lower() not in all_stopwords]
        ))

        # Perform Sentiment Analysis
        pretrained = "indonesia-bert-sentiment-classification"
        model = AutoModelForSequenceClassification.from_pretrained(pretrained)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

        def analyze_sentiment(text):
            result = sentiment_analysis(text)
            label = label_index[result[0]['label']]
            score = result[0]['score']
            return pd.Series({'sentiment_label': label, 'sentiment_score': score})

        df[['sentiment_label', 'sentiment_score']] = df['cleaned_text'].apply(analyze_sentiment)

        # Count the occurrences of each sentiment label
        sentiment_counts = df['sentiment_label'].value_counts()

        # Plot a bar chart using seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Count')
        sentiment_plot_path = 'static/sentiment_distribution.png'
        plt.savefig(sentiment_plot_path)

        model = BERTopic(verbose=True)
        model.fit(df['cleaned_text'])
        topics, probabilities = model.transform(df['cleaned_text'])
        fig = model.visualize_barchart()
        fig.write_image('static/barchart.png')
        topic_plot_path = 'static/barchart.png'

        fig1 = model.visualize_hierarchy()
        fig1.write_image('static/hierarchy.png')
        topic_plot_path1 = 'static/hierarchy.png'

        topic_distr, _ = model.approximate_distribution(df['cleaned_text'], min_similarity=0)
        fig2 = model.visualize_distribution(topic_distr[0])
        fig2.write_image('static/dist.png')
        topic_plot_path2 = 'static/dist.png'

        # Generate sentiment analysis results table
        analysis_results = df.to_html(classes='data')

        # Concatenate Cleaned text
        positive_text = ' '.join(df[df['sentiment_label'] == 'positive']['cleaned_text'])
        negative_text = ' '.join(df[df['sentiment_label'] == 'negative']['cleaned_text'])
        neutral_text = ' '.join(df[df['sentiment_label'] == 'neutral']['cleaned_text'])

        # Create WordCloud Positive
        wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='Set2', background_color='white'
        ).generate(positive_text)

        wordcloud_positive = "static/wordcloud_positive.png"
        wordcloud.to_file(wordcloud_positive)

        # Use Google Gemini API to generate content based on the uploaded image
        img = PIL.Image.open(wordcloud_positive)
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        try:
            response = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img])
            response.resolve()
            gemini_response_pos = response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_pos = "Error: Failed to generate content with Gemini API."

        # Create WordCloud Neutral
        wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='Set2', background_color='white'
        ).generate(neutral_text)

        wordcloud_neutral = "static/wordcloud_neutral.png"
        wordcloud.to_file(wordcloud_neutral)

        img = PIL.Image.open(wordcloud_neutral)
        try:
            response = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img])
            response.resolve()
            gemini_response_neu = response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neu = "Error: Failed to generate content with Gemini API."

        # Create WordCloud Negative
        wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='Set2', background_color='white'
        ).generate(negative_text)

        wordcloud_negative = "static/wordcloud_negative.png"
        wordcloud.to_file(wordcloud_negative)

        img = PIL.Image.open(wordcloud_negative)
        try:
            response = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img])
            response.resolve()
            gemini_response_neg = response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neg = "Error: Failed to generate content with Gemini API."

        # Bigram Positive
        words1 = positive_text.split()
        bigrams = list(zip(words1, words1[1:]))
        bigram_counts = collections.Counter(bigrams)
        top_bigrams = dict(bigram_counts.most_common(10))

        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
        plt.yticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=0)
        plt.xlabel('Count')
        plt.ylabel('Bigram Words')
        plt.title("Top 10 Bigram Positive Sentiment")

        bigram_positive = "static/bigram_positive.png"
        plt.savefig(bigram_positive)
        

        img1 = PIL.Image.open(bigram_positive)
        try:
            response1 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img1])
            response1.resolve()
            gemini_response_pos1 = response1.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_pos1 = "Error: Failed to generate content with Gemini API."

        # Bigram Neutral
        words2 = neutral_text.split()
        bigrams = list(zip(words2, words2[1:]))
        bigram_counts = collections.Counter(bigrams)
        top_bigrams = dict(bigram_counts.most_common(10))

        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
        plt.yticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=0)
        plt.xlabel('Count')
        plt.ylabel('Bigram Words')
        plt.title("Top 10 Bigram Neutral Sentiment")

        bigram_neutral = "static/bigram_neutral.png"
        plt.savefig(bigram_neutral)
        

        img2 = PIL.Image.open(bigram_neutral)
        try:
            response2 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img2])
            response2.resolve()
            gemini_response_neu1 = response2.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neu1 = "Error: Failed to generate content with Gemini API."

        # Bigram Negative
        words3 = negative_text.split()
        bigrams = list(zip(words3, words3[1:]))
        bigram_counts = collections.Counter(bigrams)
        top_bigrams = dict(bigram_counts.most_common(10))

        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
        plt.yticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=0)
        plt.xlabel('Count')
        plt.ylabel('Bigram Words')
        plt.title("Top 10 Bigram Negative Sentiment")

        bigram_negative = "static/bigram_negative.png"
        plt.savefig(bigram_negative)
        

        img3 = PIL.Image.open(bigram_negative)
        try:
            response3 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img3])
            response3.resolve()
            gemini_response_neg1 = response3.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neg1 = "Error: Failed to generate content with Gemini API."
            
        # Unigram Positive
        words2 = positive_text.split()

        # Count the occurrences of each word
        word_counts = collections.Counter(words2)

        # Get top 10 words
        top_words = dict(word_counts.most_common(10))

        # Create bar chart
        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_words)), list(top_words.values()), align='center')  # Horizontal bar chart
        plt.yticks(range(len(top_words)), list(top_words.keys()), rotation=0)  # Swapping y-axis and x-axis
        plt.xlabel('Count')  # Changed the label to Count
        plt.ylabel('Words')  # Changed the label to Words
        plt.title("Top 10 Unigram Positive Sentiment")
        # Save the unigram image
        unigram_positive = "static/unigram_positive.png"
        # Save the entire plot as a PNG
        plt.savefig(unigram_positive)
        # Show the plot
        

        # Use Google Gemini API to generate content based on the bigram image
        img1 = PIL.Image.open(unigram_positive)
        try:
            response1 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img1])
            response1.resolve()
            gemini_response_pos2 = response1.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_pos2 = "Error: Failed to generate content with Gemini API."
                    


        # Unigram Neutral
        words2 = neutral_text.split()

        # Count the occurrences of each word
        word_counts = collections.Counter(words2)

        # Get top 10 words
        top_words = dict(word_counts.most_common(10))

        # Create bar chart
        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_words)), list(top_words.values()), align='center')  # Horizontal bar chart
        plt.yticks(range(len(top_words)), list(top_words.keys()), rotation=0)  # Swapping y-axis and x-axis
        plt.xlabel('Count')  # Changed the label to Count
        plt.ylabel('Words')  # Changed the label to Words
        plt.title("Top 10 Unigram Neutral Sentiment")
        # Save the unigram image
        unigram_neutral = "static/unigram_neutral.png"
        # Save the entire plot as a PNG
        plt.savefig(unigram_neutral)
        # Show the plot
        

        # Use Google Gemini API to generate content based on the bigram image
        img1 = PIL.Image.open(unigram_neutral)
        try:
            response1 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img1])
            response1.resolve()
            gemini_response_neu2 = response1.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neu2 = "Error: Failed to generate content with Gemini API."




        # Unigram Negative
        words2 = negative_text.split()

        # Count the occurrences of each word
        word_counts = collections.Counter(words2)

        # Get top 10 words
        top_words = dict(word_counts.most_common(10))

        # Create bar chart
        plt.figure(figsize=(10, 10))
        plt.barh(range(len(top_words)), list(top_words.values()), align='center')  # Horizontal bar chart
        plt.yticks(range(len(top_words)), list(top_words.keys()), rotation=0)  # Swapping y-axis and x-axis
        plt.xlabel('Count')  # Changed the label to Count
        plt.ylabel('Words')  # Changed the label to Words
        plt.title("Top 10 Unigram Negative Sentiment")
        # Save the unigram image
        unigram_negative = "static/unigram_negative.png"
        # Save the entire plot as a PNG
        plt.savefig(unigram_negative)
        # Show the plot
        

        # Use Google Gemini API to generate content based on the bigram image
        img1 = PIL.Image.open(unigram_negative)
        try:
            response1 = model.generate_content([custom_question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img1])
            response1.resolve()
            gemini_response_neg2 = response1.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            gemini_response_neg2 = "Error: Failed to generate content with Gemini API."
            
        document_analyzed = True

        # Function to handle encoding to latin1
        def safe_encode(text):
            try:
                return text.encode('latin1', errors='replace').decode('latin1')  # Replace invalid characters
            except Exception as e:
                return f"Error encoding text: {str(e)}"







        # Generate PDF with the results
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.cell(200, 10, txt="Sentiment Analysis Report", ln=True, align='C')

        # Sentiment Distribution Plot
        pdf.image(sentiment_plot_path, x=10, y=30, w=190)
        pdf.ln(100)

        pdf.add_page()
        pdf.cell(200, 10, txt="Topic Modelling Barchart", ln=True, align='C')
        pdf.image(topic_plot_path, x=10, y=30, w=190)

        pdf.add_page()
        pdf.cell(200, 10, txt="Topic Modelling Hierarchy", ln=True, align='C')
        pdf.image(topic_plot_path1, x=10, y=30, w=190)

        pdf.add_page()
        pdf.cell(200, 10, txt="Topic Modelling Distribution", ln=True, align='C')
        pdf.image(topic_plot_path2, x=10, y=30, w=190)
                    

        # Positive WordCloud and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive WordCloud", ln=True, align='C')
        pdf.image(wordcloud_positive, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive WordCloud Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_pos))

        # Neutral WordCloud and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral WordCloud", ln=True, align='C')
        pdf.image(wordcloud_neutral, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral WordCloud Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neu))

        # Negative WordCloud and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative WordCloud", ln=True, align='C')
        pdf.image(wordcloud_negative, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative WordCloud Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neg))

        # Positive Bigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive Bigram Sentiment", ln=True, align='C')
        pdf.image(bigram_positive, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive Bigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_pos1))

        # Neutral Bigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral Bigram Sentiment", ln=True, align='C')
        pdf.image(bigram_neutral, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral Bigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neu1))

        # Negative Bigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative Bigram Sentiment", ln=True, align='C')
        pdf.image(bigram_negative, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative Bigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neg1))

        # Positive Unigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive Unigram Sentiment", ln=True, align='C')
        pdf.image(unigram_positive, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Positive Unigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_pos2))

        # Neutral Unigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral Unigram Sentiment", ln=True, align='C')
        pdf.image(unigram_neutral, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Neutral Unigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neu2))

        # Negative Unigram and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative Unigram Sentiment", ln=True, align='C')
        pdf.image(unigram_negative, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Negative Unigram Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(gemini_response_neg2))
        
        pdf_file_path = os.path.join("static", "output.pdf")
        pdf.output(pdf_file_path)

        pdf_file_path = os.path.join("static", "output.pdf")
        pdf_file_path = pdf_file_path.replace("\\", "/")
        

        return AnalyzeDocumentResponse(
                    meta={"status": "success", "code": 200},
                    sentiment_plot_path=sentiment_plot_path,
                    topic_plot_path=topic_plot_path, topic_plot_path1=topic_plot_path1,
                    topic_plot_path2=topic_plot_path2, 
                    wordcloud_positive=wordcloud_positive,
                    gemini_response_pos= gemini_response_pos,
                    wordcloud_neutral=wordcloud_neutral, 
                    gemini_response_neu= gemini_response_neu,
                    wordcloud_negative=wordcloud_negative,
                    gemini_response_neg= gemini_response_neg,
                    bigram_positive=bigram_positive, 
                    gemini_response_pos1= gemini_response_pos1,
                    bigram_neutral=bigram_neutral, 
                    gemini_response_neu1= gemini_response_neu1,
                    bigram_negative=bigram_negative,
                    gemini_response_neg1= gemini_response_neg1,
                    unigram_positive= unigram_positive,
                    gemini_response_pos2= gemini_response_pos2,
                    unigram_neutral= unigram_neutral,
                    gemini_response_neu2= gemini_response_neu2,
                    unigram_negative= unigram_negative,
                    gemini_response_neg2= gemini_response_neg2,
                    pdf_file_path=pdf_file_path
            )

    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))




# Route for answering questions
@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: Request,
    api_key: str = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...)
):
    global uploaded_file_path, document_analyzed, summary, api, llm
    
    loader = None

    try:

        # Initialize the LLM model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        with open(uploaded_file_path, "wb") as f:
            f.write(file.file.read())
        # Determine the file extension and select the appropriate loader
        loader = None
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()

        if file_extension == ".csv":
            loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path, mode="elements")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Load and process the document
        try:
            docs = loader.load()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")

        # Combine document text
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api_key

        # Initialize embeddings and create FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        document_search = FAISS.from_texts(chunks, embeddings)

        # Generate query embedding and perform similarity search
        query_embedding = embeddings.embed_query(question)
        results = document_search.similarity_search_by_vector(query_embedding, k=3)

        if results:
            retrieved_texts = " ".join([result.page_content for result in results])

            # Define the Summarize Chain for the question
            latest_conversation = request.cookies.get("latest_question_response", "")
            template1 = (
                f"{question} Answer the question based on the following:\n\"{text}\"\n:" +
                (f" Answer the Question with only 3 sentences. Latest conversation: {latest_conversation}" if latest_conversation else "")
            )
            prompt1 = PromptTemplate.from_template(template1)

            # Initialize the LLMChain with the prompt
            llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

            # Invoke the chain to get the summary
            try:
                response_chain = llm_chain1.invoke({"text": text})
                summary1 = response_chain["text"]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error invoking LLMChain: {str(e)}")

            # Generate embeddings for the summary
            try:
                summary_embedding = embeddings.embed_query(summary1)
                document_search = FAISS.from_texts([summary1], embeddings)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

            # Perform a search on the FAISS vector database
            try:
                if document_search:
                    query_embedding = embeddings.embed_query(question)
                    results = document_search.similarity_search_by_vector(query_embedding, k=1)

                    if results:
                        current_response = results[0].page_content
                    else:
                        current_response = "No matching document found in the database."
                else:
                    current_response = "Vector database not initialized."
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during similarity search: {str(e)}")
        else:
            current_response = "No relevant results found."

        # Append the question and response from FAISS search
        current_question = f"You asked: {question}"
        question_responses.append((current_question, current_response))

        # Save all results to output_summary.json
        save_to_json(question_responses)

        return AskResponse(meta={"status": "success", "code": 200}, question=question, result=current_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



def save_to_json(question_responses):
    outputs = {
        "question_responses": question_responses
    }
    with open("output_summary.json", "w") as outfile:
        json.dump(outputs, outfile)
    




@app.get("/download_pdf")
async def download_pdf():
    pdf_output_path = 'static/analysis_report.pdf'
    return FileResponse(pdf_output_path, filename="analysis_report.pdf", media_type='application/pdf')



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
