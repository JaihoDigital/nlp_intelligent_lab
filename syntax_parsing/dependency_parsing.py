import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

def dep_parsing(text):
    doc = nlp(text)
    results = []
    for token in doc:
        results.append(f"{token.text:15} <----{token.dep_:15} ----{token.head.text}")
    return "\n".join(results)


def visualize_deps(text):
    doc = nlp(text)
    return displacy.render(doc, style="dep", jupyter=False)

