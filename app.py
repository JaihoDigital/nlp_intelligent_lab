import streamlit as st
from utils.tokenizer import (
    word_tokenizer, sentence_tokenizer, char_tokenizer, 
    sub_word_tokenizer, ngram_tokenizer, whitespace_tokenizer, 
    regex_tokenizer, tweet_tokenizer
)
from utils.lemmatizer import single_lemmatizer, mul_lemmatizer
from utils.stemming import porterStemmer, snowballStemmer, lancasterStemmer
from utils.stopworder import stop_worder
from utils.text_norm import lower_case, upper_case, capitalize_case

from text_vectorization.bag_of_words import simple_bag_of_word
from text_vectorization.bag_of_words import advance_bow
from text_vectorization.tfidf import simple_tf_idf, advance_tf_idf

from syntax_parsing.pos_tagging import statistical_pos_tag, neural_pos_tagger
from syntax_parsing.dependency_parsing import dep_parsing, visualize_deps

from config import APP_NAME, VERSION, AUTHOR, ORG
import nltk
nltk.data.path.append('./nltk_data')


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="NLP Intelligent Lab", 
    page_icon="🧠", 
    layout="wide"
)

# ===== SIDEBAR (PHASE SELECTION) =====
with st.sidebar:
    st.header("🧭 NLP Learning Phases")

    phase = st.radio(
        "Select NLP Phase:",
        [
            "Text Preprocessing",
            "Feature Extraction",
            "Syntax & Parsing",
            "Semantic Analysis",
            "Basic ML Models & Classification",
            "Advanced Sequence Labelling",
            "Information Extraction",
            "Transformers & Modern NLP",
            "Language Generation",
            "Q/A Systems",
            "Dialogue Systems",
            "Speech Processing",
            "Sentiment & Emotional Analysis",
            "Projects"
        ],
        index=0
    )

    st.markdown("---")
    st.subheader("🔗 Resources")
    with st.expander("Papers & Docs"):
        st.markdown("""
        - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
        - [BERT Paper](https://arxiv.org/abs/1810.04805)  
        - [HuggingFace Docs](https://huggingface.co/docs)
        """)

    with st.expander("About This Lab"):
        st.markdown("""
        **Goal:** Build intuition through hands-on NLP experiments.  
        **Built With:** Streamlit ⚡  
        """)

# ===== MAIN AREA =====
st.title("🧠 NLP Intelligent Lab")
st.markdown(f"### 📘 Current Phase: {phase}")

# Define modules by phase
modules_by_phase = {
    "Text Preprocessing": [
        "Steps in TP", "Tokenizer", "Lemmatization", "Stemming",
        "Stopword Removal", "Text Normalization"
    ],
    "Feature Extraction": [
        "Bag of Words", "TF-IDF", "Word Embedding"
    ],
    "Syntax & Parsing": [
        "POS Tagging", "Dependency Parsing", "Constituency Parsing"
    ],
    "Semantic Analysis": [
        "NER", "Word Sense Disambiguation", "Coreference Resolution"
    ],
    "Basic ML Models & Classification": [
        "Text Classification (Naive Bayes, SVM)", "Sentiment Analysis (basic)", "Spam Detection"
    ],
    "Advanced Sequence Labelling": [
        "RNN", "LSTM", "GRU","CNN for Text", "Seq2Seq Models"
    ],
    "Information Extraction": [
        "Entity Extraction", "Relation Extraction", "Event Extraction"
    ],
    "Transformers & Modern NLP": [
        "BERT-GPT-T5 architecture", "Transformer-based embeddings", "Pre-trained language models"
    ],
    "Language Generation": [
        "Machine Translation", "Text Summarization", "Text Generation"
    ],
    "Q/A Systems": [
        "Retrieval-Based Q/A", "Generative Q/A"
    ],
    "Dialogue Systems": [
        "Chatbots", "Virtual Assistants"
    ],
    "Speech Processing": [
        "Speech Recognition", "Text-to-Speech (TTS)"
    ],
    "Sentiment & Emotional Analysis": [
        "Topic Modeling","Emotion Detection", "Opinion Mining"
    ],
    "Projects": [
        "Mini Projects", "Minor Projects", "Major Projects", "Capstone Projects"
    ]
}

# ===== MODULE SELECTION (on main screen) =====
st.markdown("---")
st.subheader("⚙️ Choose a Module")
current_module = st.radio(
    "Select Module for this Phase:",
    modules_by_phase[phase],
    horizontal=True
)

st.markdown("---")

# ===== MODULE LOGIC =====
# Text Preprocessing
if phase == "Text Preprocessing" and current_module == "Steps in TP":
    st.markdown("""
- **Text Normalization:** Case Normalization, Punctuation Removal, Number Handling, Whitespace, Contaction Expansion, Special Character Handling.
- **Tokenization:** Word Tokenization, Sentence Tokenization, Sub-word Tokenization, N-gram Tokenization, etc.
- **Stopword Removal:** NLTK based or custom based.
- **Stemming and Lemmatization:** Porter Stemmer, Snowball Stemmer, Lancaster Stemmer, Lemmatizer.
            """)
    st.link_button("Text Preprocessing Notes ↗","#")

elif phase == "Text Preprocessing" and current_module == "Tokenizer":
    st.subheader("📝 Text Input")
    text = st.text_area(
        "Enter your text below:",
        "Hello world! NLP Intelligent Lab is awesome!",
        height=100
    )

    # Tokenizer selection
    col1, col2 = st.columns(2)
    def tokenizer_ui(column, token_id):
        with column:
            tokenizer_option = st.selectbox(
                f"Select Tokenizer {token_id}",
                [
                    "Select...", "Word Tokenizer", "Sentence Tokenizer", 
                    "Char Tokenizer", "Sub-word Tokenizer", "N-gram Tokenizer",
                    "White Space Tokenizer", "Regex Tokenizer", "Tweet Tokenizer"
                ],
                index=0,
                key=f"tokenizer_{token_id}"
            )
            if tokenizer_option == "Select...":
                st.info("Select a tokenizer to begin.")
                return None
            
            with st.expander("⚙️ Options", expanded=False):
                if tokenizer_option == "N-gram Tokenizer":
                    n = st.number_input("N size", 1, 10, 2, key=f"ngram_{token_id}")
                elif tokenizer_option == "Regex Tokenizer":
                    pattern = st.text_input("Regex Pattern", r"\w+", key=f"regex_{token_id}")
            
            try:
                if tokenizer_option == "Word Tokenizer":
                    tokens = word_tokenizer(text)
                elif tokenizer_option == "Sentence Tokenizer":
                    tokens = sentence_tokenizer(text)
                elif tokenizer_option == "Char Tokenizer":
                    tokens = char_tokenizer(text)
                elif tokenizer_option == "Sub-word Tokenizer":
                    tokens = sub_word_tokenizer(text)
                elif tokenizer_option == "N-gram Tokenizer":
                    tokens = ngram_tokenizer(text, n)
                elif tokenizer_option == "White Space Tokenizer":
                    tokens = whitespace_tokenizer(text)
                elif tokenizer_option == "Regex Tokenizer":
                    tokens = regex_tokenizer(text, pattern)
                elif tokenizer_option == "Tweet Tokenizer":
                    tokens = tweet_tokenizer(text)
                else:
                    tokens = []

                st.success(f"✅ Token Count: {len(tokens)}")
                st.write(tokens)

                # Visualization toggle
                visualize = st.toggle("👁️ Visualize Tokens", key=f"viz_{token_id}")
                if visualize:
                    st.markdown("---")
                    st.markdown("**Visual Preview:**")
                    token_str = " | ".join([f"`{t}`" for t in tokens[:30]])
                    st.markdown(token_str)

            except Exception as e:
                st.error(f"Error: {str(e)}")

    tokenizer_ui(col1, "A")
    tokenizer_ui(col2, "B")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📘 Concept", "💻 Code", "🔗 Tokenization Other Sources"])

    with tab1:
        st.markdown("""
        ### 🔹 Tokenization
        Tokenization is the **process of breaking text into smaller units** called *tokens* — such as words, phrases, or sentences.  
        These tokens act as the **basic building blocks** for further NLP tasks like part-of-speech tagging, sentiment analysis, and named entity recognition.

        #### 🔧 Common Tokenization Techniques:
        - **Word Tokenization:** Splitting text into words (e.g., "AI is powerful" → ['AI', 'is', 'powerful'])
        - **Sentence Tokenization:** Splitting text into sentences.
        - **Character Tokenization:** Splitting text into individual characters.
        - **Subword Tokenization:** Breaking words into smaller units to handle unknown or rare words.

        Tokenization is usually the **first preprocessing step** in any NLP pipeline.
        """)

    
    with tab2:
        st.code("""
    from nltk.tokenize import word_tokenize

    # Sample text
    text = "Tokenization is the first step in NLP!"

    # Word Tokenization
    words = word_tokenize(text)
                
    print("\\nWord Tokens:")
    print(words)

    # Output:
    #
    # Word Tokens:
    # ['Tokenization', 'is', 'the', 'first', 'step', 'in', 'NLP', '!']
    """, language="python")
    
    with tab3:
        st.link_button(
            "NLTK Tokenize Docs ↗","https://www.nltk.org/api/nltk.tokenize.html"
        )
        st.link_button(
            "Spacy Tokenize Docs ↗","https://spacy.io/api/tokenizer"
        )
        st.link_button(
            "Tokenization Means ↗","https://www.geeksforgeeks.org/nlp/nlp-how-tokenizing-text-sentence-words-works/"
        )
    
    st.link_button(
        "🔗 View More Tokenization Techniques on GitHub ↗","https://github.com/avarshvir/Machine_Learning_Journey/tree/main/14_nlp/14_1_text_preprocessing/1_tokenization"
    )

elif phase == "Text Preprocessing" and current_module == "Lemmatization":
    st.subheader("📝 Text Input")
    text = st.text_area(
        "Enter your text below:",
        "I am running quickly. He went home yesterday.",
        height=100  # Better example with context!
    )

    col1, = st.columns(1)
    def lemmatizer_ui(column, lemm_id):
        with column:
            lemmatizer_option = st.selectbox(
                f"Select Lemmatizer {lemm_id}",
                ["Select...", "Single Lemmatizer (spaCy)", "Multiple Lemmatizer (spaCy)"], 
                index=0,
                key=f"lemm_{lemm_id}"
            )
            use_nltk = st.toggle("🔄 Use NLTK Fallback (better for verbs!)", key=f"nltk_{lemm_id}")
            
            if lemmatizer_option == "Select...":
                st.info("Select a lemmatizer to begin.")
                return
            
            try:
                if "Single Lemmatizer (spaCy)" in lemmatizer_option:
                    lemmas = single_lemmatizer(text, use_nltk_fallback=False)
                #elif "Single Lemmatizer (NLTK)" in lemmatizer_option:
                #    lemmas = single_lemmatizer(text, use_nltk_fallback=True)
                elif "Multiple Lemmatizer (spaCy)" in lemmatizer_option:
                    lemma_details = mul_lemmatizer(text)
                    st.success(f"✅ Processed {len(lemma_details)} tokens")
                    st.dataframe(lemma_details, use_container_width=True, hide_index=True)
                    return  # Early return for multi
                
                st.success(f"✅ Lemmas: {' | '.join(lemmas)}")
                st.json({"Original": text.split(), "Lemmas": lemmas})
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    lemmatizer_ui(col1, "-")
    #lemmatizer_ui(col2, "B")

    # ... tabs ...
    tab1, tab2, tab3 = st.tabs(["📘 Concept", "💻 Code", "🔗 Lemmatization other sources"])

    with tab1:
        st.markdown("""
        ### 🔹 Lemmatization
        **Lemmatization** reduces words to their **base/dictionary form** (lemma) considering **context/POS**.
        - "Running" (VERB) → "run"
        - "Running" (NOUN, e.g., "running business") → "running"
        - Better than stemming (no invalid words like "runn").

        #### spaCy vs NLTK:
        | Feature | spaCy | NLTK |
        |---------|--------|------|
        | Speed | ⚡ Fast (pipeline) | Slower |
        | Context | Excellent (needs sentences) | Good (needs manual POS) |
        | Verbs (isolated) | ❌ Often fails | ✅ Reliable |

        **Tip**: Use sentences for best spaCy results!
        """)

    with tab2:
        st.code("""
import spacy
from nltk.stem import WordNetLemmatizer
import nltk

nlp = spacy.load("en_core_web_sm")
nltk_lemmatizer = WordNetLemmatizer()

# spaCy (context-aware)
doc = nlp("I am running. He went home.")
lemmas = [t.lemma_ for t in doc if not t.is_punct]

# NLTK (POS-aware)
def nltk_lemma(text):
    tokens = nltk.word_tokenize(text)
    return [nltk_lemmatizer.lemmatize(w, nltk.pos_tag([w])[0][1][0].upper()) 
            for w in tokens]

print(lemmas)  # ['I', 'be', 'run', 'He', 'go', 'home']
        """, language="python")

    with tab3: 
        st.link_button("NLTK Lemmatizer Docs ↗", "https://www.nltk.org/api/nltk.stem.WordNetLemmatizer.html")
        st.link_button("Spacy Lemmatizer Docs ↗", "https://spacy.io/api/lemmatizer")
        st.link_button("Lemmatizer Means ↗", "https://www.geeksforgeeks.org/python/python-lemmatization-with-nltk/")
        

    st.link_button(
        "🔗 View More Lemmatization Techniques on GitHub ↗️", "https://github.com/avarshvir/Machine_Learning_Journey/tree/main/14_nlp/14_1_text_preprocessing/2_stemming_and_lemmatization"
    )

elif phase == "Text Preprocessing" and current_module == "Stemming":
    st.subheader("📝 Text Input")
    text = st.text_area(
        "Enter your word below:",
        "running",
        height=100  # Better example with context!
    )

    col1, = st.columns(1)
    def stemmer_ui(column, stem_id):
        with column:
            stemmer_option = st.selectbox(
                f"Select Stemmer {stem_id}",
                ["Select...", "Porter Stemmer", "Snowball Stemmer", "Lancaster Stemmer"], 
                index=0,
                key=f"lemm_{stem_id}"
            )
            #use_nltk = st.toggle("🔄 Use NLTK Fallback (better for verbs!)", key=f"nltk_{lemm_id}")
            
            if stemmer_option == "Select...":
                st.info("Select a stemmer to begin.")
                return
            
            try:
                if "Porter Stemmer" in stemmer_option:
                    p_stem = porterStemmer(text)
                    st.success(f"✅ Porter Stem: {''.join(p_stem)}")
                    st.json({"Original": text.split(), "Stem": p_stem})
                elif "Snowball Stemmer" in stemmer_option:
                    s_stem = snowballStemmer(text)
                    st.success(f"✅ Snowball Stem: {''.join(s_stem)}")
                elif "Lancaster Stemmer" in stemmer_option:
                    l_stem = lancasterStemmer(text)
                    st.success(f"✅ Lancaster Stem: {''.join(l_stem)}")
                
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    stemmer_ui(col1, "-")
    #lemmatizer_ui(col2, "B")

    # ... tabs ...
    tab1, tab2, tab3 = st.tabs(["📘 Concept", "💻 Code", "🔗 Stemming other sources"])

    with tab1:
        st.markdown("""
        ### 🔹 Stemming
        **Stemming** reduces words to their **root form** by chopping off suffixes/prefixes using rules.
        - "Running" → "run"
        - "Happily" → "happi"
        - "Going" → "go            
        - Faster than lemmatization but may produce non-dictionary words.

        #### Porter vs Snowball vs Lancaster:
        | Stemmer | Aggressiveness | Speed | Quality |
        |---------|----------------|-------|---------|
        | Porter | Moderate | Fast | Good |
        | Snowball | Moderate | Fast | Better (Porter 2.0) |
        | Lancaster | Very Aggressive | Fast | Over-stemming |

        **Tip**: Use Snowball for best balance of speed and accuracy!
        """)

    with tab2:
        st.code("""
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize

# Initialize stemmers
porter = PorterStemmer()
snowball = SnowballStemmer("english")
lancaster = LancasterStemmer()

def stem_sentence(text, stemmer):
    tokens = word_tokenize(text)
    return [stemmer.stem(token) for token in tokens]

text = "Running happily going studies"

# Compare stemmers
porter_stems = stem_sentence(text, porter)
snowball_stems = stem_sentence(text, snowball) 
lancaster_stems = stem_sentence(text, lancaster)

print("Porter:", porter_stems)    # ['run', 'happili', 'go', 'studi']
print("Snowball:", snowball_stems) # ['run', 'happili', 'go', 'studi']
print("Lancaster:", lancaster_stems) # ['run', 'happy', 'go', 'study']
""", language="python")

    with tab3: 
        st.link_button("NLTK Stemming Docs ↗", "https://www.nltk.org/howto/stem.html")
        st.link_button("Stemming Resources ↗", "https://www.analyticsvidhya.com/blog/2021/11/an-introduction-to-stemming-in-natural-language-processing/")
        st.link_button("Stemming Means ↗", "https://www.geeksforgeeks.org/machine-learning/introduction-to-stemming/")
        

    st.link_button(
        "🔗 View More Stemming Techniques on GitHub ↗️", "https://github.com/avarshvir/Machine_Learning_Journey/tree/main/14_nlp/14_1_text_preprocessing/2_stemming_and_lemmatization"
    )

elif phase == "Text Preprocessing" and current_module == "Stopword Removal":
    st.subheader("📝 Text Input")
    text = st.text_area(
        "Enter your word below:",
        "My name is Arshvir. How are you dear?",
        height=100  # Better example with context!
    )

    col1, = st.columns(1)
    def stopword_ui(column, stop_id):
        with column:
            stopword_option = st.selectbox(
                f"Select Stopword {stop_id}",
                ["Select...", "NLTK Stopword", "Custom Stopword"], 
                index=0,
                key=f"lemm_{stop_id}"
            )
            #use_nltk = st.toggle("🔄 Use NLTK Fallback (better for verbs!)", key=f"nltk_{lemm_id}")
            
            if stopword_option == "Select...":
                st.info("Select a stopword to begin.")
                return
            
            try:
                if "NLTK Stopword" in stopword_option:
                    nltk_stop = stop_worder(text)
                    st.info(f"➡️ Original Text: {text}")
                    st.success(f"✅ Stopword Removal: {''.join(nltk_stop)}")
                    st.json({"Original": text.split(), "Text": nltk_stop})
                elif "Custom Stopword" in stopword_option:
                    st.warning("Currently in progress")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    stopword_ui(col1, "-")
    #lemmatizer_ui(col2, "B")

    # ... tabs ...
    tab1, tab2, tab3 = st.tabs(["📘 Concept", "💻 Code", "🔗 Stopword Resources"])

    with tab1:
        st.markdown("""
        ### 🔹 Stopword Removal
        **Stopwords** are common words that carry little meaningful information but appear frequently in text.
        - "the", "is", "in", "and", "to", "of"
        - Removing them improves processing efficiency and focuses on important content words
        - Reduces noise in text analysis and machine learning models

        #### Common Stopword Libraries:
        | Library | Words | Languages | Use Case |
        |---------|-------|-----------|----------|
        | NLTK | 179 | English | General purpose, education |
        | spaCy | 326+ | Multiple | Production applications |
        | Scikit-learn | 318 | English | Machine learning tasks |
        | Custom | Variable | Any | Domain-specific applications |
        
        #### When to Remove Stopwords:
        ✅ **Text Classification** - Improves feature quality  
        ✅ **Topic Modeling** - Focuses on meaningful terms  
        ✅ **Search Engines** - Reduces index size  
        ❌ **Sentiment Analysis** - May remove important context words  
        ❌ **Language Translation** - Grammar structure matters  

        **Tip**: Consider your use case! Sometimes stopwords carry important contextual information.
        """)

    with tab2:
        st.code("""
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords_nltk(text):
    \"\"\"Remove stopwords using NLTK\"\"\"
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return filtered_text

def remove_stopwords_spacy(text):
    \"\"\"Remove stopwords using spaCy\"\"\"
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    filtered_text = [token.text for token in doc if not token.is_stop]
    return filtered_text

def remove_custom_stopwords(text, custom_stopwords):
    \"\"\"Remove custom defined stopwords\"\"\"
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in custom_stopwords]
    return filtered_text

# Example usage
text = "This is a sample sentence demonstrating stopword removal techniques"

# NLTK Stopwords
nltk_result = remove_stopwords_nltk(text)
print("NLTK Result:", nltk_result)

# Custom Stopwords
custom_stops = ['this', 'is', 'a', 'demonstrating']
custom_result = remove_custom_stopwords(text, custom_stops)
print("Custom Result:", custom_result)

# spaCy Stopwords (requires spaCy model)
# spacy_result = remove_stopwords_spacy(text)
# print("spaCy Result:", spacy_result)
""", language="python")

    with tab3: 
        st.link_button("NLTK Stopwords Documentation ↗", "https://www.nltk.org/howto/corpus.html#stopwords")
        st.link_button("Stopword Analysis Tutorial ↗", "https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/")
        st.link_button("Custom Stopwords Implementation ↗", "https://www.geeksforgeeks.org/removing-stop-words-nltk-python/")
        

    st.link_button(
        "🔗 View More Stopwords Techniques on GitHub ↗️", "https://github.com/avarshvir/Machine_Learning_Journey/tree/main/14_nlp/14_1_text_preprocessing/3_stopwords_removal"
    )

elif phase == "Text Preprocessing" and current_module == "Text Normalization":
    st.subheader("📝 Text Input")
    text = st.text_area(
        "Enter your word below:",
        "My name is Arshvir. How are you dear?",
        height=100  # Better example with context!
    )

    col1, = st.columns(1)
    def textnorm_ui(column, norm_id):
        with column:
            textnorm_option = st.selectbox(
                f"Select Text Normalizer {norm_id}",
                ["Select...", "Lower case", "Upper case", "Capitalize case", "Remove Punctuation", 
                 "Remove Numbers", "Remove Extra Spaces", "Advanced Cleaning"], 
                index=0,
                key=f"lemm_{norm_id}"
            )
            #use_nltk = st.toggle("🔄 Use NLTK Fallback (better for verbs!)", key=f"nltk_{lemm_id}")
            
            if textnorm_option == "Select...":
                st.info("Select a technique to begin.")
                return
            
            try:
                if "Lower case" in textnorm_option:
                    lower_text = lower_case(text)
                    st.info(f"➡️ Original Text: {text}")
                    st.success(f"✅ Lower Case Text: {''.join(lower_text)}")
                    st.json({"Original": text.split(), "Text": lower_text})
                
                elif "Upper case" in textnorm_option:
                    upper_text = upper_case(text)
                    st.info(f"➡️ Original Text: {text}")
                    st.success(f"✅ Upper Case Text: {''.join(upper_text)}")
                    st.json({"Original": text.split(), "Text": upper_text})
                
                elif "Capitalize case" in textnorm_option:
                    capitalize_text = capitalize_case(text)
                    st.info(f"➡️ Original Text: {text}")
                    st.success(f"✅ Capitalize Case Text: {''.join(capitalize_text)}")
                    st.json({"Original": text.split(), "Text": capitalize_text})
                
                    

            except Exception as e:
                st.error(f"Error: {str(e)}")

    textnorm_ui(col1, "-")
    #lemmatizer_ui(col2, "B")

    # ... tabs ...
    tab1, tab2, tab3 = st.tabs(["📘 Concept", "💻 Code", "🔗 Normalization Resources"])

    with tab1:
        st.markdown("""
        ### 🔹 Text Normalization
        **Text Normalization** converts text into a consistent, canonical form for processing.
        It's crucial for preparing text data for NLP tasks and machine learning models.
        
        #### Core Normalization Techniques:
        **1. Case Normalization**
        - **Lowercase**: "Hello World" → "hello world" (most common in NLP)
        - **Uppercase**: "important" → "IMPORTANT" (for emphasis/acronyms)
        - **Capitalize**: "john smith" → "John smith" (proper nouns)

        **2. Noise Removal**
        - **Punctuation**: "Hello!" → "Hello" (removes !, ., ,, ?, etc.)
        - **Numbers**: "Version 2.0" → "Version " (removes digits)
        - **Special Characters**: "Café" → "Caf" (removes accents/symbols)
        - **Extra Whitespace**: "Hello   world" → "Hello world"

        **3. Advanced Techniques**
        - **Accent Removal**: "naïve" → "naive"
        - **Contraction Expansion**: "I'm" → "I am"
        - **Unicode Normalization**: Standardizes character encoding

        #### Impact on NLP Tasks:
        | Task | Recommended Approach | Why |
        |------|---------------------|------|
        | Search/Retrieval | Lowercase + Punctuation Removal | Case-insensitive matching |
        | Sentiment Analysis | Keep punctuation | ! and ? carry sentiment |
        | Text Classification | Lowercase + Basic Cleaning | Consistent features |
        | Chatbots | Keep case + selective cleaning | Natural conversation |

        **Pro Tip**: Always consider your specific use case before normalizing!
        """)

    with tab2:
        st.code("""
import re
import string
import unicodedata

def to_lowercase(text):
    \"\"\"Convert text to lowercase\"\"\"
    return text.lower()

def to_uppercase(text):
    \"\"\"Convert text to uppercase\"\"\"
    return text.upper()

def capitalize_text(text):
    \"\"\"Capitalize the first character of text\"\"\"
    return text.capitalize()

def remove_punctuation(text):
    \"\"\"Remove all punctuation marks\"\"\"
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text):
    \"\"\"Remove all numerical digits\"\"\"
    return re.sub(r'\d+', '', text)

def remove_extra_spaces(text):
    \"\"\"Remove extra whitespace characters\"\"\"
    return ' '.join(text.split())

def remove_accents(text):
    \"\"\"Remove accent characters\"\"\"
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

def expand_contractions(text):
    \"\"\"Expand common English contractions\"\"\"
    contractions = {
        "don't": "do not",
        "can't": "cannot", 
        "won't": "will not",
        "i'm": "i am",
        "you're": "you are",
        "it's": "it is",
        "that's": "that is",
        "what's": "what is",
        "where's": "where is",
        "there's": "there is",
        "who's": "who is",
        "how's": "how is"
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    return text

def advanced_cleaning(text):
    \"\"\"Comprehensive text cleaning pipeline\"\"\"
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove accents
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text

# Example usage
sample_text = "Hello! I have 123 ideas and I'm REALLY excited!!!   "

print("Original:", sample_text)
print("Lowercase:", to_lowercase(sample_text))
print("No Punctuation:", remove_punctuation(sample_text))
print("No Numbers:", remove_numbers(sample_text))
print("Clean Spaces:", remove_extra_spaces(sample_text))
print("Advanced Cleaning:", advanced_cleaning(sample_text))

# Output:
# Original: Hello! I have 123 ideas and I'm REALLY excited!!!   
# Lowercase: hello! i have 123 ideas and i'm really excited!!!   
# No Punctuation: Hello I have 123 ideas and Im REALLY excited   
# No Numbers: Hello! I have  ideas and I'm REALLY excited!!!   
# Clean Spaces: Hello! I have 123 ideas and I'm REALLY excited!!!
# Advanced Cleaning: hello i have ideas and im really excited
""", language="python")

    with tab3: 
        st.link_button("Text Normalization Guide ↗", "https://www.analyticsvidhya.com/blog/2021/06/text-preprocessing-in-nlp-with-python-codes/")
        st.link_button("Advanced NLP Preprocessing ↗", "https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/")
        st.link_button("Unicode Normalization ↗", "https://towardsdatascience.com/text-normalization-for-natural-language-processing-nlp-70a314bfa646")
        st.link_button("Regular Expressions Guide ↗", "https://docs.python.org/3/howto/regex.html")

    st.link_button(
        "🔗 View More Normalization Techniques on GitHub ↗️", "https://github.com/avarshvir/Machine_Learning_Journey/tree/main/14_nlp/14_1_text_preprocessing/4_text_normalization"
    )

################################### Phase 2 ###################################
## Bags of Words
elif phase == "Feature Extraction" and current_module == "Bag of Words":
    st.subheader("📝 Text Input")
    text = st.text_area(
        "Enter your word below:",
        "I love NLP and NLP, NLP is Love",
        height=100  # Better example with context!
    )

    col1, = st.columns(1)
    def bow_ui(column, bow_id):
        with column:
            bow_option = st.selectbox(
                f"Select Vectorizer for Bag of Word {bow_id}",
                ["Select...", "Using Simple BoW", "Using Advance BoW", "Advance Techniques"], 
                index=0,
                #key=f"lemm_{stem_id}"
            )
            #use_nltk = st.toggle("🔄 Use NLTK Fallback (better for verbs!)", key=f"nltk_{lemm_id}")
            
            if bow_option == "Select...":
                st.info("Select a Bag of word technique to begin.")
                return
            
            try:
                if "Using Simple BoW" in bow_option:
                    bow_vector = simple_bag_of_word(text)
                    st.code(bow_vector, language="text")
                elif "Using Advance BoW" in bow_option:
                    advance_bow_vector = advance_bow(text)
                    st.code(advance_bow_vector, language="text")
                elif "Advance Techniques" in bow_option:
                    st.markdown("""
                    ### 🔬 Advance Techniques (In Upcoming Phases)

                    This simulator is evolving beyond basic vectorization.  
                    Soon, you'll be able to explore **deep learning-powered NLP** using:

                    - **TensorFlow & PyTorch**: Custom embeddings, attention mechanisms, and model introspection
                    - **Transformers**: Contextual tokenization, BERT-style encodings, and sentence-level semantics
                    - **Interactive Simulations**: Visualize how models interpret meaning, rank relevance, and adapt to fine-tuning

                    This isn't just a learning hub — it's a playground for experimentation.  
                    Stay tuned for modules that let you simulate, compare, and tweak real NLP pipelines.
                    """)
                


                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    bow_ui(col1, "-")

    # ... tabs ...
    tab1, tab2, tab3, tab4 = st.tabs(["📘 Concept", "💻 Code", "🔗 BoW other sources", "📝 Notes"])

    with tab1:
        st.markdown("""
        ### 🔹 Bag of Words (BoW)
        **Bag of Words** is a foundational technique in NLP that converts text into numerical vectors by counting word occurrences.
        - Ignores grammar and word order
        - Treats each word as a separate feature
        - Simple, fast, and effective for many tasks
                    
        #### Example:
        Texts:  
        - "NLP is love"  
        - "NLP is important"

        Vocabulary: ['nlp', 'is', 'love', 'important']  
        Vectors:  
        - [1, 1, 1, 0]  
        - [1, 1, 0, 1]

        #### BoW vs Advanced Techniques:
        | Technique | Context Awareness | Dimensionality | Interpretability | Use Case |
        |----------|-------------------|----------------|------------------|----------|
        | BoW | ❌ No | High | ✅ Easy | Quick prototyping |
        | TF-IDF | ❌ No | High | ✅ Easy | Keyword extraction |
        | Embeddings | ✅ Yes | Low | ❌ Hard | Deep NLP models |

        **Tip**: Use BoW for fast baselines and interpretable models. Upgrade to embeddings for semantic tasks!
        """)

    with tab2:
        st.code("""
from sklearn.feature_extraction.text import CountVectorizer

# Creating an object count vectorization
vectorizer = CountVectorizer()
                
corpus = [
    "Dog is Pet Animal and Dog is Loyal",
    "Cat and Dog are both loyal Animal"   
    ]

bow_matrix = vectorizer.fit_transform(corpus)
features = vectorizer.get_feature_names_out()

print("Features:", list(features))
print()

for i, sentence in enumerate(corpus):
    vector = bow_matrix[i].toarray()[0]
    print(f"{sentence}: {list(vector)}")
                
### Output
Features: ['and' 'animal' 'are' 'both' 'cat' 'dog' 'is' 'loyal' 'pet']
Dog is Pet Animal and Dog is Loyal: [1, 1, 0, 0, 0, 2, 2, 1, 1]
Cat and Dog are both loyal Animal: [1, 1, 1, 1, 1, 1, 0, 1, 0]


""", language="python")

    with tab3: 
        st.link_button("BoW Sklearn Explained ↗", "https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction")
        st.link_button("BoW Explained ↗", "https://www.mygreatlearning.com/blog/bag-of-words/")
        
    st.link_button(
        "🔗 View More Bag of Word Techniques on GitHub ↗️", "https://github.com/avarshvir/Machine_Learning_Journey/tree/main/14_nlp/14_2_text_vectorization/1_bag_of_words"
    )

    with tab4:
        st.markdown("""
    ### 📘 Complete Theory Notes

    **What is Text Vectorization** [Learn More](#)
                    
    #### Techniques Covered:
    - **Bag of Words (BoW)**: [Learn More](#)
    - **TF-IDF**: [Learn More](#)
    - **Embeddings**: [Learn More](#)

    These techniques form the foundation for NLP pipelines, from sentiment analysis to search engines.
    """)

## TF-IDF
elif phase == "Feature Extraction" and current_module == "TF-IDF":
    st.subheader("📝 Text Input")
    text = st.text_area(
        "Enter your word below:",
        "NLP is amazing, NLP is Love",
        height=100  # Better example with context!
    )

    col1, = st.columns(1)
    def tfidf_ui(column, tfidf_id):
        with column:
            tfidf_option = st.selectbox(
                f"Select Vectorizer for Bag of Word {tfidf_id}",
                ["Select...", "Using Simple TF-IDF", "Using Advance TF-IDF"], 
                index=0,
                #key=f"lemm_{stem_id}"
            )
            #use_nltk = st.toggle("🔄 Use NLTK Fallback (better for verbs!)", key=f"nltk_{lemm_id}")
            
            if tfidf_option == "Select...":
                st.info("Select a TFIDF vectorizer to begin.")
                return
            
            try:
                if "Using Simple TF-IDF" in tfidf_option:
                    tfidf_vector = simple_tf_idf(text)
                    st.code(tfidf_vector, language="text")
                elif "Using Advance TF-IDF" in tfidf_option:
                    advance_tfidf_vector = advance_tf_idf(text)
                    st.code(advance_tfidf_vector, language="text")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    tfidf_ui(col1, "-")

    # ... tabs ...
    tab1, tab2, tab3, tab4 = st.tabs(["📘 Concept", "💻 Code", "🔗 TF-IDF other sources", "📝 Notes"])

    with tab1:
        st.markdown("""
        ### 🔹 TF-IDF (Term Frequency-Inverse Document Frequency)
        **TF-IDF** is an advanced text vectorization technique that measures word importance by considering both frequency and rarity across documents.
        - **TF**: How often a word appears in a document (like BoW)
        - **IDF**: How rare/important a word is across all documents
        - **TF-IDF**: TF × IDF - balances frequency with uniqueness
                    
        #### Example:
        Texts:  
        - "NLP is love"  
        - "NLP is important"

        Vocabulary: ['nlp', 'is', 'love', 'important']  
        TF-IDF Vectors:  
        - [0.40, 0.40, 0.81, 0.00]  (love is unique to doc1)  
        - [0.40, 0.40, 0.00, 0.81]  (important is unique to doc2)

        #### TF-IDF vs Other Techniques:
        | Technique | Context Awareness | Common Words | Rare Words | Use Case |
        |----------|-------------------|--------------|------------|----------|
        | BoW | ❌ No | ✅ High weight | ✅ High weight | Quick prototypes |
        | **TF-IDF** | ❌ No | ❌ Penalized | ✅ Boosted | Search, classification |
        | Embeddings | ✅ Yes | ✅ Contextual | ✅ Contextual | Deep NLP tasks |

        **Tip**: Use TF-IDF when you want to emphasize unique, important words over common ones!
        """)

    with tab2:
        st.code("""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Creating TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
                    
    corpus = [
        "Dog is Pet Animal and Dog is Loyal",
        "Cat and Dog are both loyal Animal"   
    ]

    tfidf_matrix = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()

    print("Features:", list(features))
    print()

    for i, sentence in enumerate(corpus):
        vector = tfidf_matrix[i].toarray()[0]
        # Format to 2 decimal places for readability
        formatted_vector = [round(score, 2) for score in vector]
        print(f"{sentence}: {formatted_vector}")
                    
    ### Output
    Features: ['and', 'animal', 'are', 'both', 'cat', 'dog', 'is', 'loyal', 'pet']

    Dog is Pet Animal and Dog is Loyal: 
    [0.28, 0.28, 0.00, 0.00, 0.00, 0.56, 0.56, 0.28, 0.42]

    Cat and Dog are both loyal Animal: 
    [0.31, 0.31, 0.44, 0.44, 0.44, 0.31, 0.00, 0.31, 0.00]

    ### Interpretation:
    - 'dog' appears in both docs → lower scores (0.56, 0.31)
    - 'is' only in doc1 → higher score (0.56)  
    - 'cat' only in doc2 → higher score (0.44)
    - 'pet' only in doc1 → higher score (0.42)
    """, language="python")

    with tab3: 
        st.link_button("TF-IDF Sklearn Documentation ↗", "https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html")
        st.link_button("TF-IDF Explained ↗", "https://www.geeksforgeeks.org/machine-learning/understanding-tf-idf-term-frequency-inverse-document-frequency/")
        st.link_button("TF-IDF Mathematical Details ↗", "https://en.wikipedia.org/wiki/Tf–idf")

    with tab4:
        st.markdown("""
    ### 📘 Complete Theory Notes

    **What is Text Vectorization** [Learn More](#)
                    
    #### Techniques Covered:
    - **Bag of Words (BoW)**: [Learn More](#)
    - **TF-IDF**: [Learn More](#)
    - **Embeddings**: [Learn More](#)

    These techniques form the foundation for NLP pipelines, from sentiment analysis to search engines.
    """)
        
    st.link_button(
        "🔗 View More TFIDF vectorizer on GitHub ↗️", "https://github.com/avarshvir/Machine_Learning_Journey/tree/main/14_nlp/14_2_text_vectorization/2_tf_idf"
    )

################### Syntax and Parsing ###################
## POS Tagging
elif phase == "Syntax & Parsing" and current_module == "POS Tagging":
    st.subheader("📝 Text Input")
    text = st.text_area(
        "Enter your word below:",
        "The quick brown fox jumps over lazy dog",
        height=100  # Better example with context!
    )
    st.info("Spacy POS Tagger is more accurate than NLTK POS Tagger")
    col1, = st.columns(1)
    def pos_ui(column, pos_id):
        with column:
            pos_option = st.selectbox(
                
                f"Select Tagger for PoS {pos_id}",
                ["Select...", "Statistical PoS Tagger", "Neural PoS Tagger", "Tagger Types"], 
                index=0,
                #key=f"lemm_{stem_id}"
            )
            #use_nltk = st.toggle("🔄 Use NLTK Fallback (better for verbs!)", key=f"nltk_{lemm_id}")
            
            if pos_option == "Select...":
                st.info("Select a PoS tagger technique to begin.")
                return
            
            try:
                if "Statistical PoS Tagger" in pos_option:
                    spos_tagger = statistical_pos_tag(text)
                    st.code(spos_tagger, language="text")
                elif "Neural PoS Tagger" in pos_option:
                    npos_tagger = neural_pos_tagger(text)
                    st.code(npos_tagger, language="text")
                elif "Tagger Types" in pos_option:
                    st.markdown("""
                    ### 🔬 Tagger Types:
                    - Rule-based POS Tagging
                    - Statistical POS Tagging
                    - Neural POS Tagging
                    """)
                   
            except Exception as e:
                st.error(f"Error: {str(e)}")

    pos_ui(col1, "-")

    # ... tabs ...
    tab1, tab2, tab3, tab4 = st.tabs(["📘 Concept", "💻 Code", "🔗 PoS Resources", "📝 Notes"])

    with tab1:
        st.markdown("""
        ### 🔹 Part-of-Speech (PoS) Tagging
        **Part-of-Speech Tagging** is the process of marking words in a text as corresponding to particular parts of speech (nouns, verbs, adjectives, etc.) based on both definition and context.

        #### Key PoS Tags:
        | Tag | Meaning | Example |
        |-----|---------|---------|
        | NOUN | Noun | `cat`, `London`, `love` |
        | VERB | Verb | `run`, `is`, `think` |
        | ADJ | Adjective | `quick`, `beautiful` |
        | ADV | Adverb | `quickly`, `very` |
        | DET | Determiner | `the`, `a`, `this` |
        | ADP | Adposition | `in`, `on`, `at` |
        | PRON | Pronoun | `he`, `she`, `it` |
        | CONJ | Conjunction | `and`, `but`, `or` |

        #### Applications:
        - ✅ **Grammar checking** and proofreading
        - ✅ **Information extraction** (finding entities)
        - ✅ **Question answering** systems
        - ✅ **Speech recognition** and synthesis
        - ✅ **Machine translation**

        #### PoS Tagging Approaches:
        | Method | Accuracy | Speed | Use Case |
        |--------|----------|-------|----------|
        | Rule-based | Medium | Fast | Limited domains |
        | Statistical | High | Medium | General purpose |
        | Neural | Very High | Slow | High-accuracy needs |

        **Tip**: For most applications, statistical taggers (like NLTK) offer the best balance of accuracy and speed!
        """)

    with tab2:
        st.code("""
    # Method 1: Using NLTK
    import nltk
    from nltk import word_tokenize, pos_tag

    def nltk_pos_tagger(text):
        tokens = word_tokenize(text.lower())
        return pos_tag(tokens)

    text = "The quick brown fox jumps over the lazy dog"
    result = nltk_pos_tagger(text)
    print("NLTK PoS Tags:")
    for word, tag in result:
        print(f"{word:8} -> {tag}")

    # Method 2: Using spaCy (Neural)
    import spacy

    def spacy_pos_tagger(text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text.lower())
        return [(token.text, token.pos_) for token in doc]

    text = "The quick brown fox jumps over the lazy dog"
    result = spacy_pos_tagger(text)
    print("\\nspaCy PoS Tags:")
    for word, tag in result:
        print(f"{word:8} -> {tag}")

    ### Output Example:
    NLTK PoS Tags:
    the       -> DT
    quick     -> JJ
    brown     -> JJ
    fox       -> NN
    jumps     -> VBZ
    over      -> IN
    the       -> DT
    lazy      -> JJ
    dog       -> NN

    spaCy PoS Tags:
    the       -> DET
    quick     -> ADJ
    brown     -> ADJ
    fox       -> NOUN
    jumps     -> VERB
    over      -> ADP
    the       -> DET
    lazy      -> ADJ
    dog       -> NOUN
    """, language="python")

    with tab3: 
        st.link_button("NLTK PoS Tagging Documentation ↗", "https://www.nltk.org/book/ch05.html")
        st.link_button("spaCy Linguistic Features ↗", "https://spacy.io/usage/linguistic-features")
        st.link_button("Universal PoS Tags ↗", "https://universaldependencies.org/u/pos/")
        
    st.link_button(
        "🔗 View PoS Tagging Code on GitHub ↗️", 
        "https://github.com/avarshvir/Machine_Learning_Journey/tree/main/14_nlp/14_3_syntax_and_parsing/1_pos_tagging"
    )

    with tab4:
        st.markdown("""
        ### 📘 Complete PoS Tagging Notes

        #### Common PoS Tagging Challenges:
        - **Ambiguity**: Words like "run" can be noun or verb
        - **Unknown words**: Handling new/unseen vocabulary
        - **Context dependence**: "Time flies" vs "Fruit flies"

        #### Tagging Standards:
        - **Penn Treebank**: 36 tags (used by NLTK)
        - **Universal Dependencies**: 17 universal tags (used by spaCy)
        - **Brown Corpus**: 87 tags

        #### Preprocessing Tips:
        1. **Tokenize** before tagging
        2. Consider **lowercasing** for better accuracy
        3. Handle **punctuation** appropriately
        4. Use **sentence segmentation** for long texts

        #### Advanced Topics:
        - **Dependency parsing**: Relationships between words
        - **Named Entity Recognition (NER)**: Finding people, places, organizations
        - **Chunking**: Grouping adjacent words into phrases

        **Remember**: PoS tagging is the foundation for most advanced NLP tasks!
        """)

## Dependency Parsing
elif phase == "Syntax & Parsing" and current_module == "Dependency Parsing":
    st.subheader("📝 Text Input")
    text = st.text_area(
        "Enter your word below:",
        "The quick brown fox jumps over lazy dog.",
        height=100 
    )
    col1, = st.columns(1)
    def dep_ui(column, dep_id):
        with column:
            dep_option = st.selectbox(
                
                f"Select Parser for Dependency Parsing {dep_id}",
                ["Select...", "Textual Parser", "Visualize Parser"], 
                index=0,
                #key=f"lemm_{stem_id}"
            )
            if dep_option == "Select...":
                st.info("Select a Parser technique to begin.")
                return
            
            try:
                if "Textual Parser" in dep_option:
                    text_dep_parser = dep_parsing(text)
                    st.code(text_dep_parser, language="text")
                elif "Visualize Parser" in dep_option:
                    dep_html = visualize_deps(text)
                    st.components.v1.html(dep_html, height=300, scrolling=True)
 
            except Exception as e:
                st.error(f"Error: {str(e)}")

    dep_ui(col1, "-")

    # ... tabs ...
    tab1, tab2, tab3 = st.tabs(["📘 Concept", "💻 Code", "🔗 PoS Resources"])

    with tab1:
        st.markdown("""
        ### 🔹 Dependency Parser
        **Dependency Parsing** is the process of analyzing a sentence's grammatical structure by establishing relashionsip ("dependencies") between words.

        #### Applications:
        - ✅ **NER**
        - ✅ **Sentiment Analysis**
        - ✅ **Question answering** systems
        - ✅ **Text Generation**
        - ✅ **Machine translation**

        """)

    with tab2:
        st.code("""
    # Using spaCy 
    import spacy
    nlp = spacy.load("en_core_web_sm")
    sentence = "The quick brown fox jumps over the lazy dog."
    doc = nlp(sentence)
    for tokens in doc:
        print(f"{tokens.text:10} <----{tokens.dep_:10} ----{tokens.head.text}")
                
    ### Output Example:
    The        <----det        ----fox
    quick      <----amod       ----fox
    brown      <----amod       ----fox
    fox        <----nsubj      ----jumps
    jumps      <----ROOT       ----jumps
    over       <----prep       ----jumps
    the        <----det        ----dog
    lazy       <----amod       ----dog
    dog        <----pobj       ----over
    .          <----punct      ----jumps

    """, language="python")

    with tab3: 
        st.link_button("Dependency Theory ↗", "https://www.geeksforgeeks.org/compiler-design/constituency-parsing-and-dependency-parsing/")
        st.link_button("Spacy Dependency Docs ↗", "https://spacy.io/api/dependencyparser")
        
    st.link_button(
        "🔗 View Dependency Parsing Code on GitHub ↗️", 
        "https://github.com/avarshvir/Machine_Learning_Journey/tree/main/14_nlp/14_3_syntax_and_parsing/2_dependency_parsing"
    )


else:
    st.info("🚧 Module coming soon! Work in progress...")


# ===== FOOTER =====
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; font-size: 15px;'>
        <p>💡 <b>Developed by <span style="color:#1E90FF;">{AUTHOR}</span> and <span style="color:#1E90FF;">{ORG}</span></b></p>
        <p>🌐 Empowering AI, Data Science, and NLP learning through open-source innovation.</p>
        <p>⚙️ Version {VERSION} | 📅 © 2025 {AUTHOR} and {ORG} | 🧠 Maintained by the Open-Source Community</p>
        <p>
            🔗 <a href="https://github.com/avarshvir/" target="_blank" style="text-decoration:none; color:#1E90FF;">
            GitHub Repository</a> |
            📬 <a href="mailto:jaihodigital@gmail.com" style="text-decoration:none; color:#1E90FF;">
            Contact</a> |
            🌍 <a href="https://jaiho-labs.onrender.com" target="_blank" style="text-decoration:none; color:#1E90FF;">
            Official Website</a> |
            🧵 <a href="#" target="_blank" style="text-decoration:none; color:#1E90FF;">
            Important Links</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
