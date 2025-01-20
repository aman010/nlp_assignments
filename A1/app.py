#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 06:50:31 2025
"""

import streamlit as st
import json
import torch 
from Models import Glove, SkipgramNeg
import numpy as np
from preprocess import process
import streamlit.components.v1 as components

# Load the corpus and word2index
with open('A1/Model_corpus/corpus.json', 'r') as fp:
    corpus = json.load(fp)

@st.cache_data
def download_nltk_data():
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

# Call the function at the start of your app
download_nltk_data()

# Function to remove numeric tokens from a list
def pop_numeric(tokens):
    """
    Pops numeric tokens from the list and keeps only non-numeric tokens.
    """
    index = 0
    while index < len(tokens):
        if tokens[index].isdigit():  # Check if the token is numeric
            tokens.pop(index)  # Remove numeric token
        else:
            index += 1  # Only increment index if we don't pop
    return tokens

# Remove numeric tokens from the corpus
for doc in corpus:
    corpus[doc] = pop_numeric(corpus[doc])

# Load word2index mapping
word2index = np.load('A1/Model_corpus/word2index.npy', allow_pickle=True).item()

# Function to clean and preprocess the input text
def rpuncst(x):
    """
    Remove punctuation, stopwords, and stems the tokens.
    """
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    tokens = [token for token in x if token.isalnum() and token.lower() not in stop_words]
    return tokens

# Callback function for handling search
def search_callback(query, model_type):
    """
    Handles search functionality using the selected model (skipGram or Glove).
    """
    if model_type == 'skipGram':
        model = SkipgramNeg(8743, 2)
        model.load_state_dict(torch.load('A1/Model_corpus/neg_samples', weights_only=True, map_location=torch.device('cpu')))
        model.eval()  # Set the model to evaluation mode
        p = process(corpus, model, word2index)
        result = p.find_most_similar_documents(str(query), corpus)

    elif model_type == 'Glove':
        model = Glove(8743, 2)
        model.load_state_dict(torch.load('A1/Model_corpus/glove', weights_only=True))
        model.eval()  # Set the model to evaluation mode
        p = process(corpus, model, word2index)
        result = p.find_most_similar_documents(str(query), corpus)

    return result

# Streamlit app
def app():
    """Main Streamlit app function."""
    st.title("Text Search with Word2Vec, Skipgram, and Glove Models")

    # Read the HTML content from a file or as a string
    try:
        with open('A1/templates/index.html', 'r') as f:
            html_content = f.read()
        components.html(html_content, height=800, scrolling=True)
    except FileNotFoundError:
        st.warning("HTML file not found! Proceeding with basic app UI.")

    # Handle the search and model logic
    query = st.text_input("Enter your query:")
    model_type = st.selectbox("Select Model", ("skipGram", "Glove"))

    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                results = search_callback(query, model_type)
                if results:
                    st.write("Search Results:")
                    st.write(results)
                else:
                    st.warning("No results found.")
        else:
            st.error("Please enter a query to search.")

if __name__ == "__main__":
    app()
