"""
Types:
- Word Tokenize, Sentence Tokenize
- Sub Word Tokenize, Char Tokenize
- N Gram Tokenize
-----------------------------------
Techniques:
- BPE, Word Piece, Sentence Piece
-----------------------------------
Tools:
- NLTK, Regex, Spacy, Gensim, Transformers
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
import re
from nltk.tokenize import TweetTokenizer

def word_tokenizer(text):
    return word_tokenize(text)

def sentence_tokenizer(text):
    return sent_tokenize(text)

def char_tokenizer(text):
    return list(text)

def sub_word_tokenizer(text):
    s_tokenizer = RegexpTokenizer(r'\w+|un|re|ing|able')
    return s_tokenizer.tokenize(text)

def ngram_tokenizer(text, n):
    tokens = word_tokenize(text)
    return list(ngrams(tokens, int(n)))

# Other Tokenizations
def whitespace_tokenizer(text):
    return text.split()

def regex_tokenizer(text, pattern=r'\w+'):
    tokenizer = RegexpTokenizer(pattern)
    return tokenizer.tokenize(text)

def tweet_tokenizer(text):
    tokenizer = TweetTokenizer()
    return tokenizer.tokenize(text)

# Later on feature:
"""
import spacy
nlp = spacy.load("en_core_web_sm")

def spacy_tokenizer(text):
    doc = nlp(text)
    return [token.text for token in doc]
"""

"""
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

# Load or train a BPE tokenizer
def bpe_tokenizer(text):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer.encode(text).tokens

"""


