from flask import Flask, render_template, request, flash, redirect
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

app = Flask(__name__)

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


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            # Read the CSV file into a Pandas DataFrame
            df = pd.read_csv(file, delimiter=",")

            # Additional stopwords
            custom_stopwords = request.form.get('custom_stopwords', '').split(',')
            custom_stopword_list = [word.strip() for word in custom_stopwords]
            all_stopwords = data + custom_stopword_list

           # Remove hyperlinks and emoticons
            df['cleaned_text'] = df['full_text'].str.replace(hyperlink_pattern, '', regex=True)  # Remove hyperlinks
            df['cleaned_text'] = df['cleaned_text'].str.replace(emoticon_pattern, '', regex=True)  # Remove emoticons
            df['cleaned_text'] = df['cleaned_text'].str.replace(number_pattern, '', regex=True) # Remove number
            
            for stopword in custom_stopword_list:
                df['cleaned_text'] = df['cleaned_text'].str.replace(stopword, '')  

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

            # Generate sentiment analysis results table
            analysis_results = df.to_html(classes='data')

            # Concanate Cleaned text
            positive_text = ' '.join(df[df['sentiment_label'] == 'positive']['cleaned_text'])
            negative_text = ' '.join(df[df['sentiment_label'] == 'negative']['cleaned_text'])
            neutral_text = ' '.join(df[df['sentiment_label'] == 'neutral']['cleaned_text'])


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
            genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img])
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
            genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img])
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
            genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to wordcloud negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img])
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
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img1])
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
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img1])
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
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img1])
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
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram positive sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the positive sentiment analysis.", img1])
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
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img1])
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
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to unigram negative sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the negative sentiment analysis.", img1])
                response1.resolve()
                gemini_response_neg2 = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_neg2 = "Error: Failed to generate content with Gemini API."


            # Create a dictionary to store the outputs
            outputs = {
                "Sentiment Plot": sentiment_plot_path,
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



            return render_template('upload.html', sentiment_plot=sentiment_plot_path, analysis_results=analysis_results, 
                                   wordcloud_result_positive=wordcloud_positive, gemini_result_response_pos=gemini_response_pos,
                                   wordcloud_result_neutral=wordcloud_neutral, gemini_result_response_neu=gemini_response_neu,
                                   wordcloud_result_negative=wordcloud_negative, gemini_result_response_neg=gemini_response_neg,
                                   bigram_result_positive=bigram_positive, gemini_result_response_pos1=gemini_response_pos1,
                                   bigram_result_neutral=bigram_neutral, gemini_result_response_neu1=gemini_response_neu1,
                                   bigram_result_negative=bigram_negative, gemini_result_response_neg1=gemini_response_neg1,
                                   unigram_result_positive=unigram_positive, gemini_result_response_pos2=gemini_response_pos2,
                                   unigram_result_neutral=unigram_neutral, gemini_result_response_neu2=gemini_response_neu2,
                                   unigram_result_negative=unigram_negative, gemini_result_response_neg2=gemini_response_neg2)
if __name__ == '__main__':
    app.run(debug=True)








































