import os
import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS

# Sidebar
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made with by Ojabio Mesias')

def process_pdf(file):
    """ Extrae texto de un archivo PDF """
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Evitar errores si es None
    return text

def process_csv(file):
    """ Extrae texto de un archivo CSV """
    df = pd.read_csv(file)
    return df.to_string(index=False)  # Convertir el CSV en texto legible

def process_excel(file):
    """ Extrae texto de un archivo Excel """
    df = pd.read_excel(file)
    return df.to_string(index=False)

def process_xml(file):
    """ Extrae texto de un archivo XML """
    tree = ET.parse(file)
    root = tree.getroot()
    return ET.tostring(root, encoding='utf-8').decode('utf-8')

def main():
    st.header("Chat with Documents 📄📊")

    load_dotenv()

    # Subir un archivo (PDF, CSV, XML, Excel)
    uploaded_file = st.file_uploader("Upload your document", type=['pdf', 'csv', 'xml', 'xlsx', 'xls'])

    if uploaded_file is not None:
        st.write(f"📂 Archivo cargado: {uploaded_file.name}")

        # Procesar el archivo según su tipo
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            text = process_pdf(uploaded_file)
        elif file_extension == 'csv':
            text = process_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            text = process_excel(uploaded_file)
        elif file_extension == 'xml':
            text = process_xml(uploaded_file)
        else:
            st.error("❌ Formato de archivo no soportado.")
            return

        # Verificar si hay contenido en el archivo
        if not text.strip():
            st.error("⚠️ El archivo está vacío o no se pudo extraer texto.")
            return

        # Dividir el texto en fragmentos para embeddings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text=text)

        # Crear embeddings
        store_name = uploaded_file.name.rsplit('.', 1)[0]  # Quitar la extensión del archivo
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        if os.path.exists(f"{store_name}.faiss"):
            VectorStore = FAISS.load_local(store_name, embeddings)
            st.write('🔄 Embeddings cargados desde el disco')
        else:
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(store_name)

        # Input para preguntas
        query = st.text_input("Realiza una pregunta sobre tu documento")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = ChatOpenAI(model="gpt-4o-mini")
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            st.write(response)

if __name__ == '__main__':
    main()
