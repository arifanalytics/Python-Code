from flask import Flask, render_template, request, flash, redirect, send_file, url_for
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

app = Flask(__name__)
app.secret_key = 'flask_sentiment'


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

@app.route('/')
def upload_file():
    return render_template('upload.html')



@app.route('/process', methods=['POST'])
def process_file():
    global df
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_extension = os.path.splitext(uploaded_file.filename)[1]
        if file_extension == '.csv':
            uploaded_file.save('dataset.csv')
            df = pd.read_csv('dataset.csv', delimiter=",")
        elif file_extension in ['.xls', '.xlsx']:
            uploaded_file.save('dataset.xlsx')
            df = pd.read_excel('dataset.xlsx')
        else:
            return "Unsupported file format"

        # Get columns of the DataFrame
        columns = df.columns.tolist()

        return render_template('upload.html', columns=columns)
    else:
        return "No file uploaded"

@app.route('/analyze', methods=['POST'])
def analyze():
    global df
    if request.method == 'POST':
        

            # Additional stopwords
            target_variable = request.form.get('target_variable')
            if target_variable not in df.columns:
                return "Selected target variable does not exist in the dataset."

            custom_stopwords = request.form.get('custom_stopwords', '').split(',')
            custom_stopword_list = [word.strip() for word in custom_stopwords]
            all_stopwords = data + custom_stopword_list

            # Remove hyperlinks, emoticons, numbers, and stopwords
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
            # Load the sentiment analysis pipeline
            pretrained = "indonesia-bert-sentiment-classification"
            model = AutoModelForSequenceClassification.from_pretrained(pretrained)
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
            sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

            # Function to apply sentiment analysis to each row in the 'cleaned_text' column
            def analyze_sentiment(text):
                result = sentiment_analysis(text)
                label = label_index[result[0]['label']]
                score = result[0]['score']
                return pd.Series({'sentiment_label': label, 'sentiment_score': score})

            # Apply sentiment analysis to 'cleaned_text' column
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
            plt.savefig(sentiment_plot_path)  # Save the plot


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

            # Concanate Cleaned text
            positive_text = ' '.join(df[df['sentiment_label'] == 'positive']['cleaned_text'])
            negative_text = ' '.join(df[df['sentiment_label'] == 'negative']['cleaned_text'])
            neutral_text = ' '.join(df[df['sentiment_label'] == 'neutral']['cleaned_text'])


            question = request.form["custom_question"]


            # Create WordCloud Positive
            wordcloud = WordCloud(
                min_font_size=3, max_words=200, width=800, height=400,
                colormap='Set2', background_color='white'
            ).generate(positive_text)

            # Save the WordCloud image
            wordcloud_positive = "static/wordcloud_positive.png"
            wordcloud.to_file(wordcloud_positive)

            # Use Google Gemini API to generate content based on the uploaded image
            img = PIL.Image.open(wordcloud_positive)
            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response = model.generate_content([question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img])
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

            # Save the WordCloud image
            wordcloud_neutral = "static/wordcloud_neutral.png"
            wordcloud.to_file(wordcloud_neutral)

            # Use Google Gemini API to generate content based on the uploaded image
            img = PIL.Image.open(wordcloud_neutral)
            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response = model.generate_content([question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img])
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

            # Save the WordCloud image
            wordcloud_negative = "static/wordcloud_negative.png"
            wordcloud.to_file(wordcloud_negative)

            # Use Google Gemini API to generate content based on the uploaded image
            img = PIL.Image.open(wordcloud_negative)
            genai.configure(api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response = model.generate_content([question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img])
                response.resolve()
                gemini_response_neg = response.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_neg = "Error: Failed to generate content with Gemini API."



            # Bigram Positive
            # Get bigrams
            words1 = positive_text.split()
            bigrams = list(zip(words1, words1[1:]))

            # Count bigrams
            bigram_counts = collections.Counter(bigrams)

            # Get top 10 bigram
            top_bigrams = dict(bigram_counts.most_common(10))

            # Create bar chart
            plt.figure(figsize=(10, 10))
            plt.barh(range(len(top_bigrams)), list(top_bigrams.values()), align='center')  # Horizontal bar chart
            plt.yticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=0)  # Swapping y-axis and x-axis
            plt.xlabel('Count')  # Changed the label to Count
            plt.ylabel('Bigram Words')  # Changed the label to Bigram Words
            plt.title(f"Top 10 Bigram Positive Sentiment")

            # Save the Bigram image
            bigram_positive = "static/bigram_positive.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_positive)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_positive)
            try:
                response1 = model.generate_content([question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img1])
                response1.resolve()
                gemini_response_pos1 = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_pos1 = "Error: Failed to generate content with Gemini API."




            # Bigram Neutral
            # Get bigrams
            words1 = neutral_text.split()
            bigrams = list(zip(words1, words1[1:]))

            # Count bigrams
            bigram_counts = collections.Counter(bigrams)

            # Get top 10 bigram
            top_bigrams = dict(bigram_counts.most_common(10))

            # Create bar chart
            plt.figure(figsize=(10, 10))
            plt.barh(range(len(top_bigrams)), list(top_bigrams.values()), align='center')  # Horizontal bar chart
            plt.yticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=0)  # Swapping y-axis and x-axis
            plt.xlabel('Count')  # Changed the label to Count
            plt.ylabel('Bigram Words')  # Changed the label to Bigram Words
            plt.title(f"Top 10 Bigram Neutral Sentiment")

            # Save the Bigram image
            bigram_neutral = "static/bigram_neutral.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_neutral)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_neutral)
            try:
                response1 = model.generate_content([question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img1])
                response1.resolve()
                gemini_response_neu1 = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_neu1 = "Error: Failed to generate content with Gemini API."



            
            # Bigram Negative
            words1 = negative_text.split()
            bigrams = list(zip(words1, words1[1:]))

            # Count bigrams
            bigram_counts = collections.Counter(bigrams)

            # Get top 10 bigram
            top_bigrams = dict(bigram_counts.most_common(10))

            # Create bar chart
            plt.figure(figsize=(10, 10))
            plt.barh(range(len(top_bigrams)), list(top_bigrams.values()), align='center')  # Horizontal bar chart
            plt.yticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=0)  # Swapping y-axis and x-axis
            plt.xlabel('Count')  # Changed the label to Count
            plt.ylabel('Bigram Words')  # Changed the label to Bigram Words
            plt.title(f"Top 10 Bigram Negative Sentiment")

            # Save the Bigram image
            bigram_negative = "static/bigram_negative.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_negative)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_negative)
            try:
                response1 = model.generate_content([question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img1])
                response1.resolve()
                gemini_response_neg1 = response1.text
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
            plt.show()

            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(unigram_positive)
            try:
                response1 = model.generate_content([question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img1])
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
            plt.show()

            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(unigram_neutral)
            try:
                response1 = model.generate_content([question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img1])
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
            plt.show()

            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(unigram_negative)
            try:
                response1 = model.generate_content([question + "As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img1])
                response1.resolve()
                gemini_response_neg2 = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_neg2 = "Error: Failed to generate content with Gemini API."


            # Create a dictionary to store the outputs
            outputs = {
                "Sentiment Plot": sentiment_plot_path,
                "Topic Barchart Plot": topic_plot_path,
                "Topic Hierarchy Plot": topic_plot_path1,
                "Topic Hierarchy Plot": topic_plot_path2,
                "Wordcloud Positive": wordcloud_positive,
                "Gemini Wordcloud Positive": gemini_response_pos,
                "Wordcloud Neutral": wordcloud_neutral,
                "Gemini Wordcloud Neutral": gemini_response_neu,
                "Wordcloud Negative": wordcloud_negative,
                "Gemini Wordcloud Negative": gemini_response_neg,
                "Bi Gram Positive": bigram_positive,
                "Gemini BiGram Positive": gemini_response_pos1,
                "Bi Gram Neutral": bigram_neutral,
                "Gemini BiGram Neutral": gemini_response_neu1,
                "Bi Gram Negative": bigram_negative,
                "Gemini BiGram Negative": gemini_response_neg1,
                "UniGram Positive": unigram_positive,
                "Gemini UniGram Positive": gemini_response_pos2,
                "UniGram Neutral": unigram_neutral,
                "Gemini UniGram Neutral": gemini_response_neu2,
                "UniGram Negative": unigram_negative,
                "Gemini UniGram Negative": gemini_response_neg2,
            }

            # Save the dictionary as a JSON file
            with open("output.json", "w") as outfile:
                json.dump(outputs, outfile)



            





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

            pdf_output_path = 'static/analysis_report.pdf'
            pdf.output(pdf_output_path)








            return render_template('upload.html', sentiment_plot=sentiment_plot_path, topic_plot=topic_plot_path, topic_plot1=topic_plot_path1, topic_plot2=topic_plot_path2, analysis_results=analysis_results, 
                                   wordcloud_result_positive=wordcloud_positive, gemini_result_response_pos=gemini_response_pos,
                                   wordcloud_result_neutral=wordcloud_neutral, gemini_result_response_neu=gemini_response_neu,
                                   wordcloud_result_negative=wordcloud_negative, gemini_result_response_neg=gemini_response_neg,
                                   bigram_result_positive=bigram_positive, gemini_result_response_pos1=gemini_response_pos1,
                                   bigram_result_neutral=bigram_neutral, gemini_result_response_neu1=gemini_response_neu1,
                                   bigram_result_negative=bigram_negative, gemini_result_response_neg1=gemini_response_neg1,
                                   unigram_result_positive=unigram_positive, gemini_result_response_pos2=gemini_response_pos2,
                                   unigram_result_neutral=unigram_neutral, gemini_result_response_neu2=gemini_response_neu2,
                                   unigram_result_negative=unigram_negative, gemini_result_response_neg2=gemini_response_neg2)



@app.route('/download_pdf')
def download_pdf():
    pdf_output_path = 'static/analysis_report.pdf'
    return send_file(pdf_output_path, as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)







































