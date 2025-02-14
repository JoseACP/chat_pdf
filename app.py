import os
import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import google.generativeai as genai


# Cargar variables de entorno
load_dotenv(override=True)

# Configurar API de Gemini
api_key = "AIzaSyChndM4r8mNh_tAHlsfXL4jVFx3pCdi5ws"
genai.configure(api_key=api_key)

# Configurar modelo de generación
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8182,
    "response_mime_type": "text/plain"
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

# Función para extraer texto de un PDF normal
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
    return text.strip()

# Función para extraer texto de un PDF escaneado (OCR)
def pdf_to_text(pdf_path):
    pages = convert_from_path(pdf_path, 300)
    extracted_text = ""

    for i, page in enumerate(pages):
        image_path = f"temp_page_{i}.png"
        page.save(image_path, "PNG")
        text = pytesseract.image_to_string(image_path)
        extracted_text += text + "\n\n"
        os.remove(image_path)

    return extracted_text.strip()

# Función para procesar PDFs (normal o escaneado)
def process_pdf(file):
    text = extract_text_from_pdf(file)
    if not text:
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        text = pdf_to_text(temp_path)
        os.remove(temp_path)
    return text.strip()

# Función para procesar archivos CSV
def process_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)

# Función para procesar archivos Excel
def process_excel(file):
    df = pd.read_excel(file)
    return df.to_string(index=False)
# Función para procesar imágenes (JPG, PNG, TIFF)
def process_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text.strip()

# Función para procesar archivos XML
def process_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()
    return ET.tostring(root, encoding='utf-8').decode('utf-8')

# Función para enviar texto al modelo de Gemini
def analyze_text_with_gemini(text, prompt):
    response = model.generate_content([text, prompt])
    return response.text

def main():
    st.header("📄📊 Chat con Documentos")

    # Subida de archivo
    uploaded_file = st.file_uploader("Sube tu documento", type=['pdf', 'csv', 'xlsx', 'xml', 'jpg', 'png', 'tiff', 'docx'])

    if "last_file" not in st.session_state:
        st.session_state.last_file = None
        st.session_state.doc_text = None

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_file:
            # Procesar nuevo archivo
            st.session_state.last_file = uploaded_file.name
            file_extension = uploaded_file.name.split(".")[-1].lower()


#Cambiar esta seccion por switch se ve raro con tantos if
            if file_extension == "pdf":
                st.session_state.doc_text = process_pdf(uploaded_file)
            elif file_extension == "csv":
                st.session_state.doc_text = process_csv(uploaded_file)
            elif file_extension in ["xls", "xlsx"]:
                st.session_state.doc_text = process_excel(uploaded_file)
            elif file_extension == "xml":
                st.session_state.doc_text = process_xml(uploaded_file)
            elif file_extension in ["jpg", "png", "tiff"]:
                st.session_state.doc_text = process_image(uploaded_file)
            else:
                st.error("⚠️ Formato de archivo no compatible.")
                return


        if not st.session_state.doc_text:
            st.error("⚠️ No se pudo extraer texto del documento.")
            return

        st.write(f"📂 Documento cargado: {uploaded_file.name}")

        query = st.text_input("Realiza una pregunta sobre tu documento")

        if query:
            response = analyze_text_with_gemini(st.session_state.doc_text, query)
            st.write(response)

if __name__ == '__main__':
    main()
