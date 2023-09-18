import streamlit as st
from transformers import pipeline

# Load the Question Answering model
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Streamlit app title
st.title('Question Answering with BERT')

# Input for passage of text
passage = st.text_area('Enter the passage:')

# Input for question
question = st.text_input('Enter your question:')

# Button to trigger question answering
if st.button('Answer'):
    # Perform question answering using the model
    answer = qa_pipeline(question=question, context=passage)
    st.subheader('Answer:')
    st.write(answer['answer'])
