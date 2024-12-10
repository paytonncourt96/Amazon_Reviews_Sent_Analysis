import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import json
import requests
from io import BytesIO
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

sia = SentimentIntensityAnalyzer()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

@st.cache_data
def load_and_process_data():
    #streamlit did not work with datasets package so using a reduced jsonl file here
    url = "https://github.com/paytonncourt96/Amazon_Reviews_Sent_Analysis/raw/main/reduced_Movies_and_TV.jsonl.gz"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download dataset: {response.status_code}")
    
    with gzip.open(BytesIO(response.content), 'rt', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    df['sentiment_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['sentiment_analysis_rating'] = df['sentiment_score'].apply(lambda x: int(round(((x + 1) / 2) * 4 + 1)))
    df['textblob_sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['textblob_rating'] = df['textblob_sentiment'].apply(lambda x: int(round(((x + 1) / 2) * 4 + 1)))
    
    return df

def home_page():
    st.title("Home")
    st.title("Amazon Reviews Sentiment Analysis app")
    st.write("Katherine Beyer, Courtney Shammas, Onur Tekiner")
    image_url = "https://raw.githubusercontent.com/paytonncourt96/Amazon_Reviews_Sent_Analysis/main/amazon_image.png"
    st.image(image_url, caption="Amazon Reviews Sentiment Analysis", use_column_width=True)
    st.write("""
        Navigate through the sidebar to explore:
        - Methods
        - Vader Lexicon Analysis
        - TextBlob Analysis
        - Overall Performance Comparison
    """)

def decomposition():
    st.title("Methods")
    st.write("""
        **Methods used in this analysis:**
        - **Text Preprocessing**: Stopword removal, tokenization, lemmatization.
        - **Sentiment Analysis**: Utilizing Vader Lexicon and TextBlob.
    """)

def vader_lexicon(df):
    st.title("Vader Lexicon Sentiment Analysis vs. Actual Stars Given")
    actual_rating_counts = df['rating'].value_counts().sort_index()
    sentiment_analysis_counts = df['sentiment_analysis_rating'].value_counts().sort_index()
    st.bar_chart(pd.DataFrame({
        "Actual Ratings": actual_rating_counts,
        "Sentiment Ratings (Vader)": sentiment_analysis_counts
    }))

def textblob(df):
    st.title("TextBlob Sentiment Analysis vs. Actual Stars Given")
    actual_rating_counts = df['rating'].value_counts().sort_index()
    textblob_rating_counts = df['textblob_rating'].value_counts().sort_index()
    st.bar_chart(pd.DataFrame({
        "Actual Ratings": actual_rating_counts,
        "TextBlob Ratings": textblob_rating_counts
    }))

def overall_performance(df):
    st.title("Overall Performance Comparison: ")
    full_range = range(1, 6)
    actual_rating_counts = df['rating'].value_counts().reindex(full_range, fill_value=0)
    sentiment_analysis_counts = df['sentiment_analysis_rating'].value_counts().reindex(full_range, fill_value=0)
    textblob_rating_counts = df['textblob_rating'].value_counts().reindex(full_range, fill_value=0)

    indices = range(1, 6)
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - bar_width for i in indices], actual_rating_counts.values, bar_width, label="Actual Ratings")
    ax.bar(indices, sentiment_analysis_counts.values, bar_width, label="Vader Ratings")
    ax.bar([i + bar_width for i in indices], textblob_rating_counts.values, bar_width, label="TextBlob Ratings")
    ax.set_title("Overall Performance Comparison")
    ax.set_xlabel("Ratings")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

def main():
    df = load_and_process_data()

    #sidebr
    st.sidebar.title("Navigation")
    page_options = ["Home", "Methods", "Vader Lexicon", "TextBlob", "Overall Performance"]
    choice = st.sidebar.selectbox("Go to", page_options)

    if choice == "Home":
        home_page()
    elif choice == "Methods":
        decomposition()
    elif choice == "Vader Lexicon":
        vader_lexicon(df)
    elif choice == "TextBlob":
        textblob(df)
    elif choice == "Overall Performance":
        overall_performance(df)

if __name__ == "__main__":
    main()
