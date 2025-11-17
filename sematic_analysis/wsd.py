from nltk.wsd import lesk
from nltk.tokenize import word_tokenize

def word_sense_disambiguation(sentence, target_word=None):
    """
    Applies the Lesk algorithm on the given sentence.
    
    Args:
        sentence: The input sentence as string
        target_word: The word to disambiguate, if None, picks the first word in the sentence
    
    Returns:
        Tuple of (synset, definition)
    """
    words = word_tokenize(sentence)
    if not target_word:
        target_word = words[0]  # Default to the first word
    
    sense = lesk(words, target_word)
    if sense:
        return sense, sense.definition()
    else:
        return None, "No sense found"
