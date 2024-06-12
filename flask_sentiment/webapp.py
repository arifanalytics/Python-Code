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

app = Flask(__name__)

# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Create stopword remover
stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'rp', 'undang', 'pasal', 'ayat', 'bab']
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
            df = pd.read_csv(file, delimiter=";")

            # Additional stopwords
            custom_stopwords = request.form.get('custom_stopwords', '').split(',')
            custom_stopword_list = [word.strip() for word in custom_stopwords]
            all_stopwords = data + custom_stopword_list

           # Remove hyperlinks and emoticons
            df['cleaned_text'] = df['full_text'].str.replace(hyperlink_pattern, '')  # Remove hyperlinks
            df['cleaned_text'] = df['cleaned_text'].str.replace(emoticon_pattern, '')  # Remove emoticons
            df['cleaned_text'] = df['cleaned_text'].str.replace(number_pattern, '') # Remove number
            
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
            genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the wordcloud positive sentiment", img])
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
            genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the wordcloud neutral sentiment", img])
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
            genai.configure(api_key="AIzaSyB2sQh_oHbFULJ7x2vixJWAboPpPvrCKoA")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the wordcloud negative sentiment", img])
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
            plt.figure(figsize=(10, 7))
            plt.bar(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
            plt.xticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=90)
            plt.xlabel('Bigram Words')
            plt.ylabel('Count')
            plt.title(f"Top 10 Bigram Positive Sentiment")

            # Save the Bigram image
            bigram_positive = "static/bigram_positive.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_positive)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_positive)
            try:
                response1 = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the bigram positive sentiment", img1])
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
            plt.figure(figsize=(10, 7))
            plt.bar(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
            plt.xticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=90)
            plt.xlabel('Bigram Words')
            plt.ylabel('Count')
            plt.title(f"Top 10 Bigram Neutral Sentiment")

            # Save the Bigram image
            bigram_neutral = "static/bigram_neutral.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_neutral)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_neutral)
            try:
                response1 = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the bigram neutral sentiment", img1])
                response1.resolve()
                gemini_response_neu1 = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_neu1 = "Error: Failed to generate content with Gemini API."



            
            # Bigram Negative
            # Get bigrams
            words1 = negative_text.split()
            bigrams = list(zip(words1, words1[1:]))

            # Count bigrams
            bigram_counts = collections.Counter(bigrams)

            # Get top 10 bigram
            top_bigrams = dict(bigram_counts.most_common(10))

            # Create bar chart
            plt.figure(figsize=(10, 7))
            plt.bar(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
            plt.xticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=90)
            plt.xlabel('Bigram Words')
            plt.ylabel('Count')
            plt.title(f"Top 10 Bigram Negative Sentiment")

            # Save the Bigram image
            bigram_negative = "static/bigram_negative.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_negative)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_negative)
            try:
                response1 = model.generate_content(["You are a professional Data Analyst, write the complete conclusion and actionable insight based on the bigram negative sentiment", img1])
                response1.resolve()
                gemini_response_neg1 = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_neg1 = "Error: Failed to generate content with Gemini API."



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
                                   bigram_result_negative=bigram_negative, gemini_result_response_neg1=gemini_response_neg1)
if __name__ == '__main__':
    app.run(debug=True)









































