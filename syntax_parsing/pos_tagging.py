#import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy

def statistical_pos_tag(text):
    tokens = word_tokenize(text)
    return pos_tag(tokens)

def neural_pos_tagger(text):
    nlp = spacy.load("en_core_web_sm")
    # en_core_web_sm: Statistical model with CNN + linear models
    # en_core_web_trf: Transformer-based (true neural network)
    doc = nlp(text.lower())
    return [(token.text, token.pos_) for token in doc]
