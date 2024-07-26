# Sentiment Analysis with IndoBert and Google Gemini
Variable untuk GOOGLE GEMINI API KEY : api_key (disimpan dalam database ARIF)

Variable yang di input oleh user (step 1) : input file CSV

Variable yang di input oleh user (step 2) : target_variable (select target variable dari kolom yang ada dari dataset), custom_stopwords (stopwords tambahan dari user), custom_question (expected output) 

# Install IndoBert Emotion Classifier GitHub Repository on your local :

git clone https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification

# Install Pytorch for IndoBert GitHub Repository on your local :

pip --no-cache-dir install torch==2.3.0

# Install BERTopic on your local :

pip install -U bertopic

# Install kaleido :

pip install plotly kaleido

# UNTUK SENTIMENT ANALYSIS LEBIH BAIK BUAT VIRTUAL ENVIRONMENT SAJA, KARENA LIBRARY PYTORCH SANGAT BERMASALAH
