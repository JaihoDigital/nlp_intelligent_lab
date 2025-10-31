from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def stop_worder(text):
    tokens = text.split()
    final_text = [word for word in tokens if word not in stop_words]
    return " ".join(final_text)