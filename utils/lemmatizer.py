import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
import os
import sys


# NLTK Lemmatizer instance
nltk_lemmatizer = WordNetLemmatizer()

# Load spaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def get_wordnet_pos(word):
    """Map NLTK POS to WordNet POS for accurate lemmatization."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def single_lemmatizer(text, use_nltk_fallback=False):
    """
    Lemmatize ALL words in text using spaCy.
    Fallback to NLTK for better verb handling on isolated words.
    """
    if use_nltk_fallback:
        # NLTK: Handles verbs perfectly even without full context
        tokens = word_tokenize(text.lower())
        return [nltk_lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    
    # spaCy: Better with context, but weaker on isolated words
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

def mul_lemmatizer(text, show_pos=True):
    """Return detailed list: (token, lemma, POS)."""
    doc = nlp(text)
    data = [(token.text, token.lemma_, token.pos_) for token in doc if not token.is_punct and not token.is_space]
    return data