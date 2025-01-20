import streamlit as st
import streamlit.components.v1 as components
import os
import json
import numpy as np
import torch
from Models import Glove, SkipgramNeg
from A1.preprocess import process

# Set base path for file management
BASE_PATH = os.path.dirname(__file__)  # Directory of the script
HTML_PATH = os.path.join(BASE_PATH, 'A1/templates/index.html')  # Adjust path as needed

# Read and render the HTML file
with open('A1/templates/index.html', 'r') as f:
    html_content = f.read()

# Display the HTML file using Streamlit's components
components.html(html_content, height=800)

# Now, handle the search logic in Streamlit

# Example function to handle search and results
@st.cache_data
def load_models():
    # This is a dummy function to simulate model loading (adjust based on your actual model logic)
    word2index = np.load('A1/Model_corpus/word2index.npy', allow_pickle=True).item()
    return word2index

word2index = load_models()  # Load necessary models or data here

def search_callback(query, model_type):
    try:
        # Load model based on selected model type
        if model_type == 'skipGram':
            model = SkipgramNeg(8743, 2)
            model.load_state_dict(torch.load('A1/Model_corpus/neg_samples', weights_only=True))
            model.eval()
            p = process(corpus, model, word2index)
            result = p.find_most_similar_documents(query, corpus)

        elif model_type == 'Glove':
            model = Glove(8743, 2)
            model.load_state_dict(torch.load('A1/Model_corpus/glove', weights_only=True))
            model.eval()
            p = process(corpus, model, word2index)
            result = p.find_most_similar_documents(query, corpus)

        return result
    except Exception as e:
        st.error(f"Error in search callback: {str(e)}")
        return {}

# Streamlit app
def app():
    st.title("Text Search with Word2Vec, Skipgram, and Glove Models")

    # Create text input and model selector in Streamlit
    query = st.text_input("Enter your query:")
    model_type = st.selectbox("Select Model", ("skipGram", "Glove"))

    if st.button("Search"):
        if query:
            # Perform search callback
            results = search_callback(query, model_type)

            # Display results in Streamlit (dynamically update after AJAX call)
            if results:
                st.write("Search Results:")
                st.write(results)
            else:
                st.write("No results found.")
        else:
            st.error("Please enter a query to search.")

if __name__ == "__main__":
    app()
