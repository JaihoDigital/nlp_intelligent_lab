from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize


def porterStemmer(text):
    txt = word_tokenize(text)
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in txt]
    return " ".join(stemmed_words)

def snowballStemmer(text):
    txt = word_tokenize(text)
    ss = SnowballStemmer("english")
    stemmed_words = [ss.stem(w) for w in txt]
    return " ".join(stemmed_words)

def lancasterStemmer(text):
    txt = word_tokenize(text)
    ls = LancasterStemmer()
    stemmed_words = [ls.stem(w) for w in txt]
    return " ".join(stemmed_words)