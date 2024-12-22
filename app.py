import streamlit as st
import joblib  # Replace pickle with joblib
import numpy as np
import re
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences

# Load the saved model and tokenizer using joblib
model = joblib.load('model.joblib')  # Changed extension to .joblib
tokenizer = joblib.load('tokenizer.joblib')

# Preprocessing function
def preprocess_text(text):
    TAG_RE = re.compile(r'<[^>]+>')
    stop_words = set(stopwords.words('english'))

    # Remove HTML tags
    text = TAG_RE.sub('', text)
    # Lowercase text
    text = text.lower()
    # Remove punctuations and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text

# Streamlit UI
st.title('Sentiment Analysis App')

# Input box for user review
review = st.text_area('Enter your review:')

if st.button('Predict Sentiment'):
    if not review:
        st.error('Please enter a review to analyze!')
    else:
        # Preprocess the review
        processed_review = preprocess_text(review)
        sequences = tokenizer.texts_to_sequences([processed_review])
        padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

        # Predict sentiment
        prediction = model.predict(padded)
        sentiment = "Positive" if prediction[0] >= 0.5 else "Negative"

        # Display result
        st.success(f'Sentiment: {sentiment}')
