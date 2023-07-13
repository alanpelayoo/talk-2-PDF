import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
import textwrap
from pytube import YouTube
from openai_helper import *


def state_clean():
    #Clean state cb function
    for key in st.session_state.keys():
           del st.session_state[key]

st.set_page_config(page_title="talk-2-PDF", page_icon="🔍", layout="wide")

with st.container():
        st.title("Tu asistente PDF ")
        st.write("Chatea con tu PDF, sube tu archivo y preguntale lo que quieras 💬")
        st.write(
                """
                **Pasos:**
            1. Sube tu archivo PDF.
            2. Dale click en analizar PDF.
            3. Haz las preguntas que quieras acerca de tu PDF.

                """
        )

st.sidebar.subheader("Parámetros")
temperature = st.sidebar.slider('Temperatura: Controla la creatividad de la respuesta, 0 significa una respuesta muy directa, mientras que 1 sera una respuesta muy aletoria y creativa.', 0.0, 1.0, 0.0)
max_length = st.sidebar.slider('La longitud máxima de tokens a utilizar en la respuesta', 0, 700, 350)

# create a file upload component and prompt the user to upload a PDF file
file = st.file_uploader("Sube el archivo PDF", type="pdf", on_change=state_clean)

if file is not None:
    if "df" not in st.session_state:
        
        # read the contents of the uploaded file using PyPDF2
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

    st.success(f"Archivo subido {file.name}")

    if st.button("Analizar PDF"):
        with st.spinner("Analizando PDF..."):
            try:
                context_embeddings = compute_doc_embeddings(st.session_state["df"])
                df_embeds = pd.DataFrame(context_embeddings).transpose()
                st.session_state["df_embeds"] = df_embeds
                st.success("PDF analizado")
            except:
                st.error('Una disculpa, estamos teniendo problemas en estos momentos, pronto quedaran solucionados...', icon="🚨")
                 
    if "df_embeds" in st.session_state:
        with st.container():
            st.write("---")
            st.subheader("Pregunta lo que quieras")
            label = "Introduce tu pregunta"
            prompt = st.text_input(label, key="prompt_input", placeholder=None)
            temperature = temperature
            max_length = max_length
            
            if st.button("Pregunta"):
                with st.spinner("..."):
                    document_embeddings, new_df = generate_newdfs(st.session_state["df_embeds"], st.session_state["df"])
                    prompt_response = answer_query_with_context(prompt, new_df, document_embeddings, False, temperature, max_length)
                    st.success("Ok ..")
                    st.write(prompt_response)

footer="""<style>
a:link , a:visited{
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
text-align: center;
color: gray;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a href="https://twitter.com/realapcodes" target="_blank">apcodes</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)