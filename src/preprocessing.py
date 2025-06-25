import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess(path):
    # Load the dataset (TSV format with no header)
    df = pd.read_csv(path, sep='\t', header=None, names=['label', 'message'])

    # Convert labels to 0 (ham) and 1 (spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Clean the text messages
    df['message'] = df['message'].apply(clean_text)

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['message'])
    y = df['label'].values

    # Save vectorizer for Streamlit and deployment
    os.makedirs("models", exist_ok=True)
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
