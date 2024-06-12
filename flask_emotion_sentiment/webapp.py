from flask import Flask, render_template, request, flash, redirect
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import PIL.Image
from wordcloud import WordCloud
import collections
import json
import google.generativeai as genai

app = Flask(__name__)

# Create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Create stopword remover
stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia', 'bahwa', 'oleh', 'rp', 'undang', 'pasal', 'ayat', 'bab']
data = stop_factory.get_stop_words() + more_stopword

# Define patterns for removal
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
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file:
            df = pd.read_csv(file, delimiter=",")

            custom_stopwords = request.form.get('custom_stopwords', '').split(',')
            custom_stopword_list = [word.strip() for word in custom_stopwords]
            all_stopwords = data + custom_stopword_list

            df['cleaned_text'] = df['full_text'].str.replace(hyperlink_pattern, '')
            df['cleaned_text'] = df['cleaned_text'].str.replace(emoticon_pattern, '')
            df['cleaned_text'] = df['cleaned_text'].str.replace(number_pattern, '')
            
            for stopword in custom_stopword_list:
                df['cleaned_text'] = df['cleaned_text'].str.replace(stopword, '')  

            df['cleaned_text'] = df['cleaned_text'].apply(lambda x: ' '.join(
                [stemmer.stem(word) for word in stop_factory.create_stop_word_remover().remove(x).split()
                if word.lower() not in all_stopwords]
            ))

            from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

            tokenizer = BertTokenizer.from_pretrained("indobert-emotion-classification")
            config = BertConfig.from_pretrained("indobert-emotion-classification")
            model = BertForSequenceClassification.from_pretrained("indobert-emotion-classification", config=config)
            from transformers import pipeline

            nlp = pipeline("text-classification", model="indobert-emotion-classification")
            results = df['cleaned_text'].apply(lambda x: nlp(x)[0])
            df['label'] = [res['label'] for res in results]
            df['score'] = [res['score'] for res in results]

            sentiment_counts = df['label'].value_counts()

            sns.set(style="whitegrid")
            plt.figure(figsize=(8, 6))
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
            plt.title('Sentiment Distribution')
            plt.xlabel('Sentiment Label')
            plt.ylabel('Count')
            sentiment_plot_path = 'static/sentiment_distribution.png'
            plt.savefig(sentiment_plot_path)

            analysis_results = df.to_html(classes='data')

            anger_text = ' '.join(df[df['label'] == 'Anger']['cleaned_text'])
            happy_text = ' '.join(df[df['label'] == 'Happy']['cleaned_text'])
            neutral_text = ' '.join(df[df['label'] == 'Neutral']['cleaned_text'])
            fear_text = ' '.join(df[df['label'] == 'Fear']['cleaned_text'])
            sadness_text = ' '.join(df[df['label'] == 'Sadness']['cleaned_text'])
            love_text = ' '.join(df[df['label'] == 'Love']['cleaned_text'])

            # Bigram Anger
            # Get bigrams
            words1 = anger_text.split()
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
            plt.title(f"Top 10 Bigram Anger Sentiment")

            # Save the Bigram image
            bigram_anger = "static/bigram_anger.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_anger)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_anger)
            genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram anger sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the anger sentiment analysis.", img1])
                response1.resolve()
                gemini_response_anger = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_anger = "Error: Failed to generate content with Gemini API."



            # Bigram Happy
            # Get bigrams
            words1 = happy_text.split()
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
            plt.title(f"Top 10 Bigram Happy Sentiment")

            # Save the Bigram image
            bigram_happy = "static/bigram_happy.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_happy)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_happy)
            genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram happy sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the happy sentiment analysis.", img1])
                response1.resolve()
                gemini_response_happy = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_happy = "Error: Failed to generate content with Gemini API."


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
            genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram neutral sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the neutral sentiment analysis.", img1])
                response1.resolve()
                gemini_response_neu = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_neu = "Error: Failed to generate content with Gemini API."



            # Bigram fear
            # Get bigrams
            words1 = fear_text.split()
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
            plt.title(f"Top 10 Bigram Fear Sentiment")

            # Save the Bigram image
            bigram_fear = "static/bigram_fear.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_fear)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_fear)
            genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram fear sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the fear sentiment analysis.", img1])
                response1.resolve()
                gemini_response_fear = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_fear = "Error: Failed to generate content with Gemini API."




            # Bigram sadness
            # Get bigrams
            words1 = sadness_text.split()
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
            plt.title(f"Top 10 Bigram Sadness Sentiment")

            # Save the Bigram image
            bigram_sadness = "static/bigram_sadness.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_sadness)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_sadness)
            genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram sadness sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the sadness sentiment analysis.", img1])
                response1.resolve()
                gemini_response_sadness = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_sadness = "Error: Failed to generate content with Gemini API."

            
            # Bigram love
            words1 = love_text.split()
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
            plt.title(f"Top 10 Bigram Love Sentiment")

            # Save the Bigram image
            bigram_love = "static/bigram_love.png"
            # Save the entire plot as a PNG
            plt.savefig(bigram_love)
            plt.show()


            # Use Google Gemini API to generate content based on the bigram image
            img1 = PIL.Image.open(bigram_love)
            genai.configure(api_key="AIzaSyC0HGxZs1MI5Nfc_9v9C9b5b7vTSMSlITc")  # Replace with your API key
            model = genai.GenerativeModel('gemini-pro-vision')

            try:
                response1 = model.generate_content(["As a marketing consultant, I aim to analyze consumer insights derived from the chart and the current market context. By focusing on the key findings related to bigram love sentiment, I can formulate actionable insights. Please provide explanations in bullet points based on the love sentiment analysis.", img1])
                response1.resolve()
                gemini_response_love = response1.text
            except Exception as e:
                print(f"Error generating content with Gemini: {e}")
                gemini_response_love = "Error: Failed to generate content with Gemini API."



            # Create a dictionary to store the outputs
            outputs = {
                "Sentiment Plot": sentiment_plot_path,
                "Bi Gram Anger": bigram_anger,
                "Gemini BiGram Anger": gemini_response_anger,
                "Bi Gram Happy": bigram_happy,
                "Gemini BiGram Happy": gemini_response_happy,
                "Bi Gram Neutral": bigram_neutral,
                "Gemini BiGram Neutral": gemini_response_neu,
                "Bi Gram Fear": bigram_fear,
                "Gemini BiGram Fear": gemini_response_fear,
                "Bi Gram Sadness": bigram_sadness,
                "Gemini BiGram Sadness": gemini_response_sadness,
                "Bi Gram Love": bigram_love,
                "Gemini BiGram Love": gemini_response_love,
            }

            with open("output.json", "w") as outfile:
                json.dump(outputs, outfile)

            return render_template('upload.html', sentiment_plot=sentiment_plot_path, analysis_results=analysis_results, 
                                    bigram_result_anger=bigram_anger, gemini_result_response_anger=gemini_response_anger,
                                    bigram_result_happy=bigram_happy, gemini_result_response_happy=gemini_response_happy,
                                    bigram_result_neutral=bigram_neutral, gemini_result_response_neu=gemini_response_neu,
                                    bigram_result_fear=bigram_fear, gemini_result_response_fear=gemini_response_fear,
                                    bigram_result_sadness=bigram_sadness, gemini_result_response_sadness=gemini_response_sadness,
                                    bigram_result_love=bigram_love, gemini_result_response_love=gemini_response_love)
        
if __name__ == '__main__':
    app.run(debug=True)
