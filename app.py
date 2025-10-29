import streamlit as st
from utils.tokenizer import (
    word_tokenizer, sentence_tokenizer, char_tokenizer, 
    sub_word_tokenizer, ngram_tokenizer, whitespace_tokenizer, 
    regex_tokenizer, tweet_tokenizer
)
from config import APP_NAME, VERSION, AUTHOR, ORG
import nltk
import time
from collections import Counter
nltk.data.path.append('./nltk_data')


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="NLP Intelligent Lab", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .token-box {
        display: inline-block;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        border: 2px solid;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .token-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .phase-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
        margin: 5px;
    }
    .freq-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 25px;
        border-radius: 5px;
        margin: 5px 0;
        display: flex;
        align-items: center;
        padding-left: 10px;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===== SIDEBAR (PHASE SELECTION) =====
with st.sidebar:
    st.markdown("# 🧭 NLP Learning Hub")
    
    # Progress tracker
    if 'completed_modules' not in st.session_state:
        st.session_state.completed_modules = set()
    
    phase = st.radio(
        "🎯 Select NLP Phase:",
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
    
    # Quick Stats
    with st.expander("📊 Your Progress", expanded=False):
        modules_by_phase = {
            "Text Preprocessing": ["Tokenizer", "Lemmatization", "Stemming", "Stopword Removal", "Text Normalization"],
            "Syntax & Parsing": ["POS Tagging", "Dependency Parsing", "Constituency Parsing"],
            "Semantic Analysis": ["NER", "Word Sense Disambiguation", "Coreference Resolution"],
            "Information Extraction": ["Entity Extraction", "Relation Extraction"],
            "Text Classification & Sequence Labelling": ["Sentiment Analysis", "Topic Modeling", "Spam Detection"],
            "Feature Extraction": ["Bag of Words / TF-IDF", "Word Embedding"],
            "Advanced Sequence Modeling & Deep Learning": ["N-grams", "Seq2Seq", "RNN / LSTM / GRU", "CNN", "Transformers"],
            "Language Generation": ["Machine Translation", "Text Summarization", "Text Generation"],
            "Speech Processing": ["Speech Recognition", "Text-to-Speech (TTS)"],
            "Q/A Systems": ["Retrieval-Based Q/A", "Generative Q/A"],
            "Dialogue Systems": ["Chatbots", "Virtual Assistants"],
            "Sentiment & Emotional Analysis": ["Emotion Detection", "Opinion Mining"]
        }
        
        total_modules = sum(len(modules) for modules in modules_by_phase.values())
        completed = len(st.session_state.completed_modules)
        progress = completed / total_modules if total_modules > 0 else 0
        st.progress(progress)
        st.metric("Modules Completed", f"{completed}/{total_modules}")
    
    st.subheader("🔗 Resources")
    with st.expander("📚 Papers & Docs"):
        st.markdown("""
        - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)  
        - [BERT Paper](https://arxiv.org/abs/1810.04805)  
        - [HuggingFace Docs](https://huggingface.co/docs)
        - [NLTK Documentation](https://www.nltk.org/)
        """)

    with st.expander("ℹ️ About This Lab"):
        st.markdown(f"""
        **🎯 Goal:** Build intuition through hands-on NLP experiments  
        **⚡ Built With:** Streamlit + NLTK  
        **👨‍💻 Version:** {VERSION}  
        **🔧 Features:**
        - Interactive tokenization
        - Visual comparisons
        - Real-time statistics
        - Export capabilities
        """)

# ===== MAIN AREA =====
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title("🧠 NLP Intelligent Lab")
with col_badge:
    st.markdown(f"<div class='phase-badge'>📘 {phase}</div>", unsafe_allow_html=True)

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

# ===== MODULE SELECTION =====
st.markdown("---")
current_module = st.selectbox(
    "⚙️ Choose a Module for this Phase:",
    modules_by_phase[phase]
)

st.markdown("---")

# ===== MODULE LOGIC =====
if phase == "Text Preprocessing" and current_module == "Tokenizer":
    
    # Sample texts
    sample_texts = {
        "Default": "Hello world! NLP Intelligent Lab is awesome! 🚀",
        "Tweet": "Just launched my new #NLP project! 🎉 Check it out @ https://example.com @username",
        "Code Snippet": "def tokenize(text): return text.split()",
        "Poetry": "The woods are lovely, dark and deep. But I have promises to keep.",
        "Technical": "The TF-IDF vectorizer transforms text to feature vectors (n=10,000).",
        "Multilingual": "Hello, 你好, مرحبا, Bonjour! Welcome to NLP 🌍"
    }
    
    # Text input with examples
    col_input, col_examples = st.columns([3, 1])
    
    with col_input:
        st.subheader("📝 Text Input")
        text = st.text_area(
            "Enter your text below:",
            sample_texts["Default"],
            height=120,
            help="Type or paste any text you want to tokenize"
        )
    
    with col_examples:
        st.subheader("💡 Quick Examples")
        selected_example = st.selectbox(
            "Load example:",
            list(sample_texts.keys())
        )
        if st.button("📥 Load Example", use_container_width=True):
            text = sample_texts[selected_example]
            st.rerun()

    # Tokenizer comparison mode
    st.markdown("---")
    comparison_mode = st.checkbox("🔄 Enable Comparison Mode", value=False)
    
    num_cols = 2 if comparison_mode else 1
    cols = st.columns(num_cols)
    
    # Store tokenizer results
    tokenizer_results = []
    
    def tokenizer_ui(column, token_id, col_index):
        with column:
            st.markdown(f"### 🔧 Tokenizer {token_id}")
            
            tokenizer_option = st.selectbox(
                f"Select Tokenizer",
                [
                    "Word Tokenizer", "Sentence Tokenizer", 
                    "Char Tokenizer", "Sub-word Tokenizer", "N-gram Tokenizer",
                    "White Space Tokenizer", "Regex Tokenizer", "Tweet Tokenizer"
                ],
                index=0 if col_index == 0 else 1,
                key=f"tokenizer_{token_id}"
            )
            
            # Options
            n, pattern = 2, r"\w+"
            with st.expander("⚙️ Advanced Options", expanded=False):
                if tokenizer_option == "N-gram Tokenizer":
                    n = st.slider("N-gram size", 1, 5, 2, key=f"ngram_{token_id}")
                elif tokenizer_option == "Regex Tokenizer":
                    pattern = st.text_input("Regex Pattern", r"\w+", key=f"regex_{token_id}",
                                          help="e.g., r'\\w+' for words, r'\\d+' for numbers")
            
            try:
                # Tokenization with timing
                start_time = time.time()
                
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
                
                elapsed_time = (time.time() - start_time) * 1000  # in ms
                
                # Statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                unique_tokens = len(set(tokens))
                
                with col_stat1:
                    st.metric("🔢 Token Count", len(tokens))
                with col_stat2:
                    st.metric("🎯 Unique Tokens", unique_tokens)
                with col_stat3:
                    st.metric("⚡ Speed", f"{elapsed_time:.2f}ms")
                
                # Vocabulary richness
                if len(tokens) > 0:
                    richness = (unique_tokens / len(tokens)) * 100
                    st.progress(richness / 100)
                    st.caption(f"📊 Vocabulary Richness: {richness:.1f}%")
                
                # Token visualization
                viz_option = st.radio(
                    "Visualization:",
                    ["List View", "Colorful Tokens", "Frequency Chart"],
                    horizontal=True,
                    key=f"viz_{token_id}"
                )
                
                if viz_option == "List View":
                    st.write(tokens[:100])  # Show first 100
                    if len(tokens) > 100:
                        st.info(f"Showing first 100 of {len(tokens)} tokens")
                
                elif viz_option == "Colorful Tokens":
                    colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C', '#FFE4B5']
                    html_tokens = []
                    for i, token in enumerate(tokens[:50]):  # Show first 50
                        color = colors[i % len(colors)]
                        escaped_token = str(token).replace('<', '&lt;').replace('>', '&gt;')
                        html_tokens.append(
                            f'<span class="token-box" style="background-color:{color}; border-color:{color};" '
                            f'title="Position: {i}, Length: {len(str(token))}">{escaped_token}</span>'
                        )
                    st.markdown("".join(html_tokens), unsafe_allow_html=True)
                    if len(tokens) > 50:
                        st.info(f"Showing first 50 of {len(tokens)} tokens")
                
                elif viz_option == "Frequency Chart":
                    if len(tokens) > 0:
                        freq = Counter(tokens)
                        top_10 = freq.most_common(10)
                        
                        st.markdown("**Top 10 Most Frequent Tokens:**")
                        
                        # Find max frequency for scaling
                        max_freq = top_10[0][1] if top_10 else 1
                        
                        for token, count in top_10:
                            width_percent = (count / max_freq) * 100
                            st.markdown(
                                f'<div class="freq-bar" style="width: {width_percent}%;">'
                                f'{token}: {count}</div>',
                                unsafe_allow_html=True
                            )
                
                # Export options
                with st.expander("💾 Export Tokens", expanded=False):
                    # Plain text export
                    tokens_text = "\n".join([str(t) for t in tokens])
                    st.download_button(
                        "📄 Download as Text File",
                        tokens_text,
                        f"tokens_{token_id}.txt",
                        "text/plain",
                        use_container_width=True
                    )
                    
                    # Statistics export
                    stats_text = f"""Token Statistics for {tokenizer_option}
================================
Total Tokens: {len(tokens)}
Unique Tokens: {unique_tokens}
Vocabulary Richness: {richness:.2f}%
Processing Time: {elapsed_time:.2f}ms

Top 10 Frequent Tokens:
"""
                    freq = Counter(tokens)
                    for token, count in freq.most_common(10):
                        stats_text += f"{token}: {count}\n"
                    
                    st.download_button(
                        "📊 Download Statistics",
                        stats_text,
                        f"stats_{token_id}.txt",
                        "text/plain",
                        use_container_width=True
                    )
                
                # Store results for comparison
                tokenizer_results.append({
                    'name': tokenizer_option,
                    'tokens': tokens,
                    'count': len(tokens),
                    'unique': unique_tokens,
                    'time': elapsed_time
                })
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try adjusting your input or tokenizer settings")
    
    # Create tokenizer UIs
    for i, col in enumerate(cols):
        token_id = chr(65 + i)  # A, B, C...
        tokenizer_ui(col, token_id, i)
    
    # Comparison view
    if comparison_mode and len(tokenizer_results) == 2:
        st.markdown("---")
        st.markdown("### 🔍 Tokenizer Comparison")
        
        result_a, result_b = tokenizer_results
        
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        with col_comp1:
            diff = result_b['count'] - result_a['count']
            st.metric(
                "Token Count Difference",
                diff,
                delta=f"{abs(diff)} tokens",
                delta_color="off"
            )
        
        with col_comp2:
            diff_unique = result_b['unique'] - result_a['unique']
            st.metric(
                "Unique Tokens Difference",
                diff_unique,
                delta=f"{abs(diff_unique)} unique",
                delta_color="off"
            )
        
        with col_comp3:
            speed_diff = result_b['time'] - result_a['time']
            faster = 'B faster' if speed_diff < 0 else 'A faster'
            st.metric(
                "Speed Difference",
                f"{abs(speed_diff):.2f}ms",
                delta=faster,
                delta_color="off"
            )
        
        # Common and unique tokens
        set_a = set(result_a['tokens'])
        set_b = set(result_b['tokens'])
        common = set_a & set_b
        only_a = set_a - set_b
        only_b = set_b - set_a
        
        col_venn1, col_venn2, col_venn3 = st.columns(3)
        
        with col_venn1:
            st.metric("🤝 Common Tokens", len(common))
            with st.expander("View samples"):
                st.write(list(common)[:10])
        
        with col_venn2:
            st.metric(f"🅰️ Only in {result_a['name']}", len(only_a))
            with st.expander("View samples"):
                st.write(list(only_a)[:10])
        
        with col_venn3:
            st.metric(f"🅱️ Only in {result_b['name']}", len(only_b))
            with st.expander("View samples"):
                st.write(list(only_b)[:10])

    st.markdown("---")
    
    # Educational content
    tab1, tab2, tab3 = st.tabs(["📘 Concept", "💻 Code Examples", "🎓 Quiz"])

    with tab1:
        st.markdown("""
        ## 🔹 Understanding Tokenization
        
        Tokenization is the **process of breaking text into smaller units** called *tokens*. These tokens serve as 
        the **foundation** for all NLP tasks.
        
        ### 🎯 Why Tokenization Matters
        
        - **Feature Engineering**: Converts unstructured text into analyzable units
        - **Model Input**: Most NLP models require tokenized input
        - **Language Understanding**: Helps identify linguistic patterns
        
        ### 🔧 Tokenization Techniques
        
        **Word Tokenizer**  
        *Use Case:* General NLP tasks  
        *Example:* "Hello world" → ['Hello', 'world']
        
        **Sentence Tokenizer**  
        *Use Case:* Document segmentation  
        *Example:* "Hi! How are you?" → ['Hi!', 'How are you?']
        
        **Character Tokenizer**  
        *Use Case:* Character-level models  
        *Example:* "Cat" → ['C', 'a', 't']
        
        **Subword Tokenizer**  
        *Use Case:* Handling rare words  
        *Example:* "unhappiness" → ['un', 'happiness']
        
        **N-gram Tokenizer**  
        *Use Case:* Capturing context  
        *Example:* "I love NLP" → ['I love', 'love NLP']
        
        **Tweet Tokenizer**  
        *Use Case:* Social media text  
        *Example:* "#AI is cool! 😊" → ['#AI', 'is', 'cool', '!', '😊']
        
        ### 💡 Best Practices
        
        1. **Choose based on task**: Different tokenizers for different purposes
        2. **Handle edge cases**: Punctuation, URLs, emojis
        3. **Consider language**: Some languages need special tokenizers
        4. **Think about vocabulary size**: Affects model complexity
        """)

    with tab2:
        st.code("""
# 1. Basic Word Tokenization
from nltk.tokenize import word_tokenize

text = "Tokenization is the first step in NLP!"
tokens = word_tokenize(text)
print(tokens)
# Output: ['Tokenization', 'is', 'the', 'first', 'step', 'in', 'NLP', '!']

# 2. Sentence Tokenization
from nltk.tokenize import sent_tokenize

text = "Hello world! How are you? NLP is amazing."
sentences = sent_tokenize(text)
print(sentences)
# Output: ['Hello world!', 'How are you?', 'NLP is amazing.']

# 3. Tweet Tokenizer (handles hashtags, mentions, emojis)
from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer()
text = "This is #awesome! @user check it out 😊"
tokens = tknzr.tokenize(text)
print(tokens)
# Output: ['This', 'is', '#awesome', '!', '@user', 'check', 'it', 'out', '😊']

# 4. Regex Tokenizer (custom patterns)
from nltk.tokenize import RegexpTokenizer

# Extract only words (no punctuation)
tokenizer = RegexpTokenizer(r'\\w+')
text = "Hello, world! This is NLP."
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['Hello', 'world', 'This', 'is', 'NLP']

# 5. WhitespaceTokenizer
from nltk.tokenize import WhitespaceTokenizer

tokenizer = WhitespaceTokenizer()
text = "This is   a    test"
tokens = tokenizer.tokenize(text)
print(tokens)
# Output: ['This', 'is', 'a', 'test']
""", language="python")
        
        st.link_button(
            "🔗 View More Examples on GitHub ↗",
            "https://github.com/avarshvir/Machine_Learning_Journey/tree/main/14_nlp/14_1_text_preprocessing/1_tokenization",
            use_container_width=True
        )

    with tab3:
        st.markdown("### 🎓 Test Your Knowledge")
        
        q1 = st.radio(
            "**Q1: Which tokenizer is best for handling social media text with hashtags and mentions?**",
            ["Word Tokenizer", "Tweet Tokenizer", "Sentence Tokenizer", "Character Tokenizer"],
            key="quiz1"
        )
        if q1 == "Tweet Tokenizer":
            st.success("✅ Correct! Tweet Tokenizer is specifically designed for social media text.")
        elif q1:
            st.error("❌ Try again! Think about which tokenizer handles special symbols.")
        
        q2 = st.radio(
            "**Q2: What is the main advantage of subword tokenization?**",
            ["Faster processing", "Handles unknown/rare words better", "Produces fewer tokens", "Works only in English"],
            key="quiz2"
        )
        if q2 == "Handles unknown/rare words better":
            st.success("✅ Correct! Subword tokenization breaks rare words into known subunits.")
        elif q2:
            st.error("❌ Not quite! Think about vocabulary coverage.")
        
        q3 = st.radio(
            "**Q3: Why is tokenization usually the first step in NLP pipelines?**",
            ["It makes text shorter", "It converts text into processable units", "It removes all punctuation", "It translates text"],
            key="quiz3"
        )
        if q3 == "It converts text into processable units":
            st.success("✅ Perfect! Tokenization breaks text into units that can be analyzed.")
        elif q3:
            st.error("❌ Not quite! Think about the fundamental purpose.")
        
        if st.button("🎉 Mark Module as Complete"):
            st.session_state.completed_modules.add(f"{phase}-{current_module}")
            st.balloons()
            st.success("Great job! Module marked as complete. 🏆")

else:
    st.info("🚧 Module coming soon! Work in progress...")
    st.markdown("""
    ### 🔜 What's Coming Next?
    
    We're actively developing modules for:
    - 🌿 Lemmatization & Stemming
    - 🏷️ POS Tagging with interactive visualization
    - 🎯 Named Entity Recognition with live examples
    - 🗑️ Stopword Removal
    - 📝 Text Normalization
    - And much more!
    
    **Want to contribute?** Check out our GitHub repository!
    """)


# ===== FOOTER =====
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; font-size: 14px; padding: 20px; background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%); border-radius: 10px;'>
        <p style='font-size: 18px; margin-bottom: 10px;'>
            💡 <b>Developed by <span style="color:#667eea;">{AUTHOR}</span> & <span style="color:#764ba2;">{ORG}</span></b>
        </p>
        <p style='margin: 5px 0;'>🌐 Empowering AI, Data Science, and NLP learning through open-source innovation</p>
        <p style='margin: 5px 0;'>⚙️ Version {VERSION} | 📅 © 2025 | 🧠 Open-Source Community</p>
        <p style='margin-top: 15px;'>
            🔗 <a href="https://github.com/avarshvir/" target="_blank" style="text-decoration:none; color:#667eea; font-weight: bold;">
            GitHub</a> &nbsp;|&nbsp;
            📬 <a href="mailto:jaihodigital@gmail.com" style="text-decoration:none; color:#667eea; font-weight: bold;">
            Contact</a> &nbsp;|&nbsp;
            🌍 <a href="https://jaiho-labs.onrender.com" target="_blank" style="text-decoration:none; color:#667eea; font-weight: bold;">
            Website</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)