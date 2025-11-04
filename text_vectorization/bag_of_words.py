from sklearn.feature_extraction.text import CountVectorizer

def advance_bow(text_input):
    if ',' in text_input:
        texts = [t.strip() for t in text_input.split(',') if t.strip()]
    else:
        texts = [text_input.strip()]

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1,1),
        max_features=100
    )
    bow_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    matrix_array = bow_matrix.toarray()

    result_bow = []
    result_bow.append("Features: " + str(feature_names.tolist()) + "\n")
    for sentence, vector in zip(texts, matrix_array):
        result_bow.append(f"{sentence}: {vector.tolist()}")
    
    return "\n".join(result_bow)

"""ef simple_bag_of_word(texts):
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    matrix_array = bow_matrix.toarray()

    result_bow = []
    result_bow.append("Features: " + str(feature_names.tolist()) + "\n")
    for sentence, vector in zip(texts, matrix_array):
        result_bow.append(f"{sentence}: {vector.tolist()}")
    
    return "\n".join(result_bow)"""


def simple_bag_of_word(text_input):
    if ',' in text_input:
        texts = [t.strip() for t in text_input.split(',') if t.strip()]
    else:
        texts = [text_input.strip()]

    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    matrix_array = bow_matrix.toarray()

    result_bow = []
    result_bow.append("Features: " + str(feature_names.tolist()) + "\n")
    for sentence, vector in zip(texts, matrix_array):
        result_bow.append(f"{sentence}: {vector.tolist()}")
    
    return "\n".join(result_bow)




