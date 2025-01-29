import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_openai import ChatOpenAI
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

def main():
    st.header("Chat with PDF 📄")

    load_dotenv()

    # Subir un PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        st.write(pdf.name)  # Solo mostrar si el PDF existe

        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Evitar errores si es None

        # Dividir el texto en fragmentos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text=text)

        # Crear embeddings
        store_name = pdf.name[:-4]
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        if os.path.exists(f"{store_name}.faiss"):
            VectorStore = FAISS.load_local(store_name, embeddings)
            st.write('Embeddings cargados desde el disco')
        else:
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            VectorStore.save_local(store_name)

        # Input para preguntas
        query = st.text_input("Realiza una pregunta sobre tu PDF")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = ChatOpenAI(model_name='gpt-4o-mini')
            chain = load_qa_chain(llm=llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

            st.write(response)

if __name__ == '__main__':
    main()
