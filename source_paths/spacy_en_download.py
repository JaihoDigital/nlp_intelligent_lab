import spacy

nlp = spacy.load("en_core_web_sm")

text = "better"
doc = nlp(text)
print(doc[0].lemma_)