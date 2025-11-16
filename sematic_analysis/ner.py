import spacy
import pandas as pd
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
pd.set_option("display.max_rows", 200)

def ner_analysis(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_, ent.lemma_, ent.start_char, ent.end_char) for ent in doc.ents]
    df = pd.DataFrame(entities, columns=['text', 'type', 'lemma', 'start_pos', 'end_pos'])
    
    # Add entity descriptions
    entity_descriptions = {
        'PERSON': 'People, including fictional',
        'NORP': 'Nationalities or religious or political groups',
        'FAC': 'Buildings, airports, highways, bridges, etc.',
        'ORG': 'Companies, agencies, institutions, etc.',
        'GPE': 'Countries, cities, states',
        'LOC': 'Non-GPE locations, mountain ranges, bodies of water',
        'PRODUCT': 'Objects, vehicles, foods, etc. (not services)',
        'EVENT': 'Named hurricanes, battles, wars, sports events, etc.',
        'WORK_OF_ART': 'Titles of books, songs, etc.',
        'LAW': 'Named documents made into laws',
        'LANGUAGE': 'Any named language',
        'DATE': 'Absolute or relative dates or periods',
        'TIME': 'Times smaller than a day',
        'PERCENT': 'Percentage, including "%"',
        'MONEY': 'Monetary values, including unit',
        'QUANTITY': 'Measurements, as of weight or distance',
        'ORDINAL': '"first", "second", etc.',
        'CARDINAL': 'Numerals that do not fall under another type'
    }
    
    df['description'] = df['type'].map(entity_descriptions)
    
    html_output = displacy.render(doc, style="ent", jupyter=False)
    return df, html_output