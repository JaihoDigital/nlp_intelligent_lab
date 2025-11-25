import  pickle
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Path to models folder relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load vectorizer
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

# Load model
with open(os.path.join(MODEL_DIR, "spam_model.pkl"), "rb") as f:
    model = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict_email(text):
    ct = clean_text(text)
    vector = tfidf.transform([ct])
    pred = model.predict(vector)[0]
    return "Spam" if pred == 1 else "Ham"