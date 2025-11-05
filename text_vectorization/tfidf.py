from sklearn.feature_extraction.text import TfidfVectorizer

def simple_tf_idf(text_input):
    if ',' in text_input:
        texts = [t.strip() for t in text_input.split(',') if t.strip()]
    else:
        texts = [text_input.strip()]

    t_vectorizer = TfidfVectorizer()
    tfidf_matrix = t_vectorizer.fit_transform(texts)
    tfidf_feature_names = t_vectorizer.get_feature_names_out()
    tfidf_matrix_array = tfidf_matrix.toarray()

    result_tfid = []
    result_tfid.append("Features: " + str(tfidf_feature_names.tolist()) + "\n")
    for sentence, vector in zip(texts, tfidf_matrix_array):
        result_tfid.append(f"{sentence}: {vector.tolist()}")
    
    return "\n".join(result_tfid)

def advance_tf_idf(text_input):
    if ',' in text_input:
        texts = [t.strip() for t in text_input.split(',') if t.strip()]
    else:
        texts = [text_input.strip()]

    t_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1,1),
        max_features=100
    )
    tfidf_matrix = t_vectorizer.fit_transform(texts)
    tfidf_feature_names = t_vectorizer.get_feature_names_out()
    tfidf_matrix_array = tfidf_matrix.toarray()

    result_tfidf = []
    result_tfidf.append("Features: " + str(tfidf_feature_names.tolist()) + "\n")
    for sentence, vector in zip(texts, tfidf_matrix_array):
        result_tfidf.append(f"{sentence}: {vector.tolist()}")
    
    return "\n".join(result_tfidf)