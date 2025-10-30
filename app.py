import streamlit as st
from utils.tokenizer import (
    word_tokenizer, sentence_tokenizer, char_tokenizer, 
    sub_word_tokenizer, ngram_tokenizer, whitespace_tokenizer, 
    regex_tokenizer, tweet_tokenizer
)
from utils.lemmatizer import single_lemmatizer, mul_lemmatizer
from utils. stemming import porterStemmer, snowballStemmer, lancasterStemmer
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
            "Syntax & Parsing",
            "Semantic Analysis",
            "Information Extraction",
            "Text Classification & Sequence Labelling",
            "Feature Extraction",
            "Advanced Sequence Modeling & Deep Learning",
            "Language Generation",
            "Speech Processing",
            "Q/A Systems",
            "Dialogue Systems",
            "Sentiment & Emotional Analysis"
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
        "Tokenizer", "Lemmatization", "Stemming",
        "Stopword Removal", "Text Normalization"
    ],
    "Syntax & Parsing": [
        "POS Tagging", "Dependency Parsing", "Constituency Parsing"
    ],
    "Semantic Analysis": [
        "NER", "Word Sense Disambiguation", "Coreference Resolution"
    ],
    "Information Extraction": [
        "Entity Extraction", "Relation Extraction"
    ],
    "Text Classification & Sequence Labelling": [
        "Sentiment Analysis", "Topic Modeling", "Spam Detection"
    ],
    "Feature Extraction": [
        "Bag of Words / TF-IDF", "Word Embedding"
    ],
    "Advanced Sequence Modeling & Deep Learning": [
        "N-grams", "Seq2Seq", "RNN / LSTM / GRU", "CNN", "Transformers"
    ],
    "Language Generation": [
        "Machine Translation", "Text Summarization", "Text Generation"
    ],
    "Speech Processing": [
        "Speech Recognition", "Text-to-Speech (TTS)"
    ],
    "Q/A Systems": [
        "Retrieval-Based Q/A", "Generative Q/A"
    ],
    "Dialogue Systems": [
        "Chatbots", "Virtual Assistants"
    ],
    "Sentiment & Emotional Analysis": [
        "Emotion Detection", "Opinion Mining"
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
if phase == "Text Preprocessing" and current_module == "Tokenizer":
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
