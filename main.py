import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def home_page():
    st.title("Amazon Review Sentiment Analysis")
    st.write("Group Members: Katherine Beyer, Courtney Shammas, Onur Tekiner")
    st.write("""
        Navigate through the sidebar to explore the analysis methods,
        sentiment analysis results using Vader Lexicon, TextBlob, and
        the overall performance comparison.
    """)
    image_url = 'https://github.com/paytonncourt96/Amazon_Reviews_Sent_Analysis/raw/main//Amazon_image.png'
    st.image(image_url,  width=600, use_column_width=False)

def main():
  st.sidebar.title("Navigation")
  page_options = ["Home", "Methods", "Vader Lexicon", "Textblob", "Overall Performance"]
  choice = st.sidebar.selectbox("Go to", page_options)

  if choice == "Home":
    home_page()
  elif choice == "Methods":
    decomposition()
  elif choice == "Textblob":
    textblob()
  elif choice == "Vader Lexicon
    vader_lexicon()
  elif choice == "Overall Performance":
    overall_performance()

if __name__ == "__main__":
    main()

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
  dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Movies_and_TV", trust_remote_code=True)
  df = pd.DataFrame(dataset["full"][:10000])
  df = df[['text', 'rating']]

  df['cleaned_text'] = df['text'].apply(preprocess_text)

  #Sent Analysis
  df['sentiment_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
  df['textblob_sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

  ##changing sentiment to range 1-5 to compare to actual stars given
  def sentiment_to_star(score):
    return int(round(((score + 1) / 2) * 4 + 1))

  df['sentiment_analysis_rating'] = df['sentiment_score'].apply(sentiment_to_star)
  df['textblob_rating'] = df['textblob_sentiment'].apply(sentiment_to_star)

  return df

def decomposition():
  st.title("Methods")
  st.write("""
  - **Text Preprocessing
  - **Feature Extraction**
  - **Sentiment Analysis**: We applied Vader Lexicon and Textblob to extract sentiment scores.
  """)

def vader_lexicon():
  st.title("Vader Lexicon Setiment Analysis")
  st.write("Here we analyze sentiment vs. star reviews using Vader Lexicon.")
  actual_rating_counts = df['rating'].value_counts().sort_index()
  sentiment_analysis_counts = df['sentiment_analysis_rating'].value_counts().sort_index()
  st.bar_chart(pd.DataFrame({
      "Actual Ratings": actual_rating_counts,
      "Sentiment Ratings (Vader)": sentiment_analysis_counts}))

def textblob():
  st.title("Textblob Setiment Analysis")
  st.write("Here we analyze sentiment vs. star reviews using Textblob.")
  actual_rating_counts = df['rating'].value_counts().sort_index()
  textblob_rating_counts = df['textblob_rating'].value_counts().sort_index()
  st.bar_chart(pd.DataFrame({
      "Actual Ratings": actual_rating_counts,
      "TextBlob Ratings": textblob_rating_counts}))

def overall_performance():
  st.title("Overall Performance")
  st.write("Here we compare the performance of Vader Lexicon and Textblob vs. actual stars given in the rating.")
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