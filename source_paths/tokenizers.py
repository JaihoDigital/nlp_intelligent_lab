import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

text = "Hi, Arsh I am Arsh :)"
words = word_tokenize(text)

print(words)
