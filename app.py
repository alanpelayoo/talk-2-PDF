import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import textwrap
import pytube
import whisper
import datetime
from pytube import YouTube
from transformers import GPT2TokenizerFast
from openai_helper import *

st.set_page_config(page_title="talk-2-PDF", page_icon="🔍", layout="wide")

with st.container():
        st.title("Tu asistente PDF ")
        st.write("Chatea con tu PDF, sube tu archivo y preguntale lo que quieras 💬")
        st.write(
        """
        **Pasos:**
    1. Sube tu archivo PDF.
    2. Dale click en analizar PDF.
    3. Preguntale lo que quieras al PDF.

        """)

st.sidebar.subheader("Parámetros")
temperature = st.sidebar.slider('Temperatura: Controla la creatividad de la respuesta, 0 significa una respuesta muy directa, mientras que 1 sera una respuesta muy aletoria y creativa.', 0.0, 1.0, 0.5)
max_length = st.sidebar.slider('La longitud máxima de tokens a utilizar en la respuesta', 0, 1000, 500)

# create a file upload component and prompt the user to upload a PDF file
file = st.file_uploader("Sube el archivo PDF file", type="pdf")

if file is not None:
    if "df" not in st.session_state:
        # read the contents of the uploaded file using PyPDF2
        print("setting state")
        pdf_reader = PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
        # split the text into paragraphs
        #paragraphs = text.split('\n')
        paragraphs = textwrap.wrap(text, width=500)
        
        # create a DataFrame with a single column 'content' containing each paragraph as a separate row
        data = [{'content': paragraph} for paragraph in paragraphs]
        df = pd.DataFrame(data)
        # add a 'title' column to the DataFrame
        df['title'] = "Upload"
        # add a 'heading' column to the DataFrame
        df['heading'] = range(1, len(df) + 1)
        # add a 'tokens' column to the DataFrame
        df['tokens'] = df['content'].str.len() / 4
        df = df[df['tokens'] >= 5]
        df = df.reindex(columns=['title', 'heading', 'content', 'tokens'])
        st.session_state["df"] = df

    st.success("Archivo subido")
    # create a button to compute embeddings
    
    if st.button("Analizar PDF"):
        with st.spinner("Analizando PDF..."):
            # Perform the time-consuming computation
            context_embeddings = compute_doc_embeddings(st.session_state["df"])
            df_embeds = pd.DataFrame(context_embeddings).transpose()
            st.session_state["df_embeds"] = df_embeds
        # Replace the loading message with a success message
            
        st.success("PDF analizado")

    
    with st.container():
        st.write("---")
        st.subheader("Ask questions about the document")
        label = "Enter your question here"
        prompt = st.text_input(label, key="prompt_input", placeholder=None)
        temperature = temperature
        max_length = max_length
        if st.button("Generate answer"):
            document_embeddings, new_df = load_embeddings('sample_embeddings.csv', 'sample_df.csv')
            
            prompt_response = answer_query_with_context(prompt, new_df, document_embeddings, False, temperature, max_length)
            st.success("MLQ response")
            st.write(prompt_response)
            print(document_embeddings)
            st.write(new_df)

            new_df2 = load_embeddings2(st.session_state["df_embeds"],st.session_state["df"])
            st.write(new_df2)
            