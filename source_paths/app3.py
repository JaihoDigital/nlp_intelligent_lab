import streamlit as st
from utils.tokenizer import (word_tokenizer, sentence_tokenizer, 
                             char_tokenizer, sub_word_tokenizer, ngram_tokenizer,
                             whitespace_tokenizer, regex_tokenizer, tweet_tokenizer)

st.set_page_config(page_title="NLP Intelligent Lab",page_icon="🧠🧪",layout="wide")

st.title("NLP Intelligent Lab")

# Input
text = st.text_area("Enter text:", "Hello world! This is a simple tokenizer.")

st.markdown("---")
st.markdown("#### Tokenizer")
# Layout columns
col1, col2 = st.columns(2)

with col1:
    tokenizer_option1 = st.selectbox("Select Tokenizer (Left)",
                        ("Select...","Word Tokenizer", "Sentence Tokenizer", "Char Tokenizer", "Sub-word Tokenizer", "N-gram Tokenizer",
                         "White Space Tokenizer", "Regex Tokenizer", "Tweet Tokenizer"),
                        index=0
                    )
    if tokenizer_option1 != "Select...":
        if tokenizer_option1 == 'Word Tokenizer':
            words = word_tokenizer(text)
            st.write("**Words:**", words)
        elif tokenizer_option1 == 'Sentence Tokenizer':
            sentences = sentence_tokenizer(text)
            st.write("**Sentences:**", sentences)
        elif tokenizer_option1 == 'Char Tokenizer':
            char = char_tokenizer(text)
            st.write("**Char Tokenizer:**", char)
        elif tokenizer_option1 == 'Sub-word Tokenizer':
            sub_words = sub_word_tokenizer(text)
            st.write("**Sub-Word Tokenizer:**", sub_words)
        elif tokenizer_option1 == 'N-gram Tokenizer':
            n_size = st.number_input("Enter N-Gram Size", min_value=1, max_value=10, value=2, step=1)
            n_grams = ngram_tokenizer(text, int(n_size))
            #n_grams = ngram_tokenizer(text, st.number_input("Enter Number of size"))
            st.write("**N-Gram Tokenization:**", n_grams)
        elif tokenizer_option1 == 'White Space Tokenizer':
            whitespace_word = whitespace_tokenizer(text)
            st.write("**White Space:**", whitespace_word)
        elif tokenizer_option1 == 'Regex Tokenizer':
            st.markdown("### Regex Tokenizer")
            # Preset pattern selector
            preset = st.selectbox("Choose a preset pattern", [
            "Custom", "Words (\\w+)", "Numbers (\\d+)", "Hashtags (#\\w+)", 
            "Mentions (@\\w+)", "Punctuation ([^\\w\\s])", "4-letter words (\\b\\w{4}\\b)"])
            # Map preset to actual regex
            pattern_map = {
            "Words (\\w+)": r"\w+",
            "Numbers (\\d+)": r"\d+",
            "Hashtags (#\\w+)": r"#\w+",
            "Mentions (@\\w+)": r"@\w+",
            "Punctuation ([^\\w\\s])": r"[^\w\s]",
            "4-letter words (\\b\\w{4}\\b)": r"\b\w{4}\b"}

            # If "Custom", show text input
            if preset == "Custom":
                pattern = st.text_input("Enter your custom regex pattern", value=r"\w+")
            else:
                pattern = pattern_map[preset]
            # Tokenize and display
            if pattern:
                regex_tokens = regex_tokenizer(text, pattern)
                st.write("**Regex Tokens:**", regex_tokens)

        elif tokenizer_option1 == 'Tweet Tokenizer':
            tweet_text = tweet_tokenizer(text)
            st.write(tweet_text)
            
        
    
    # Add other tokenizer logic here...

with col2:
    tokenizer_option2 = st.selectbox("Select Tokenizer (Right)",
                        ("Select...","Word Tokenizer", "Sentence Tokenizer", "Char Tokenizer", "Sub-word Tokenizer", "N-gram Tokenizer",
                         "White Space Tokenizer", "Regex Tokenizer", "Tweet Tokenizer"),
                        index=0
                    )
    if tokenizer_option1 != "Select...":
        if tokenizer_option2 == 'Word Tokenizer':
            words = word_tokenizer(text)
            st.write("**Words:**", words)
        elif tokenizer_option2 == 'Sentence Tokenizer':
            sentences = sentence_tokenizer(text)
            st.write("**Sentences:**", sentences)
        elif tokenizer_option2 == 'Char Tokenizer':
            char = char_tokenizer(text)
            st.write("**Char Tokenizer:**", char)
        elif tokenizer_option1 == 'Sub-word Tokenizer':
            sub_words = sub_word_tokenizer(text)
            st.write("**Sub-Word Tokenizer:**", sub_words)
        elif tokenizer_option1 == 'N-gram Tokenizer':
            n_size = st.number_input("Enter N-Gram Size", min_value=1, max_value=10, value=2, step=1)
            n_grams = ngram_tokenizer(text, int(n_size))
            #n_grams = ngram_tokenizer(text, st.number_input("Enter Number of size"))
            st.write("**N-Gram Tokenization:**", n_grams)
        elif tokenizer_option1 == 'White Space Tokenizer':
            whitespace_word = whitespace_tokenizer(text)
            st.write("**White Space:**", whitespace_word)
        elif tokenizer_option1 == 'Regex Tokenizer':
            st.markdown("### Regex Tokenizer")
            # Preset pattern selector
            preset = st.selectbox("Choose a preset pattern", [
            "Custom", "Words (\\w+)", "Numbers (\\d+)", "Hashtags (#\\w+)", 
            "Mentions (@\\w+)", "Punctuation ([^\\w\\s])", "4-letter words (\\b\\w{4}\\b)"])
            # Map preset to actual regex
            pattern_map = {
            "Words (\\w+)": r"\w+",
            "Numbers (\\d+)": r"\d+",
            "Hashtags (#\\w+)": r"#\w+",
            "Mentions (@\\w+)": r"@\w+",
            "Punctuation ([^\\w\\s])": r"[^\w\s]",
            "4-letter words (\\b\\w{4}\\b)": r"\b\w{4}\b"}

            # If "Custom", show text input
            if preset == "Custom":
                pattern = st.text_input("Enter your custom regex pattern", value=r"\w+")
            else:
                pattern = pattern_map[preset]
            # Tokenize and display
            if pattern:
                regex_tokens = regex_tokenizer(text, pattern)
                st.write("**Regex Tokens:**", regex_tokens)

        elif tokenizer_option1 == 'Tweet Tokenizer':
            tweet_text = tweet_tokenizer(text)
            st.write(tweet_text)
        # Add other tokenizer logic here...
