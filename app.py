import os
import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import google.generativeai as genai
from io import BytesIO

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

# Límite de tamaño de archivo (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Función para extraer texto de un PDF normal
@st.cache_data
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
    return text.strip()

# Función para extraer texto de un PDF escaneado (OCR)
@st.cache_data
def pdf_to_text(pdf_bytes):
    images = convert_from_bytes(pdf_bytes, dpi=200)  # Reducir DPI para mayor velocidad
    extracted_text = ""

    for page in images:
        text = pytesseract.image_to_string(page, lang='spa')  # Especificar idioma
        extracted_text += text + "\n\n"

    return extracted_text.strip()

# Función para procesar PDFs (normal o escaneado)
@st.cache_data
def process_pdf(file):
    text = extract_text_from_pdf(file)
    if not text:
        pdf_bytes = file.getvalue()
        text = pdf_to_text(pdf_bytes)
    return text.strip()

# Función para procesar archivos CSV
@st.cache_data
def process_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)

# Función para procesar archivos Excel
@st.cache_data
def process_excel(file):
    df = pd.read_excel(file)
    return df.to_string(index=False)

# Función para procesar imágenes (JPG, PNG, TIFF)
@st.cache_data
def process_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image, lang='spa')
    return text.strip()

# Función para procesar archivos XML
@st.cache_data
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
    uploaded_file = st.file_uploader("Sube tu documento", type=['pdf', 'csv', 'xlsx', 'xml', 'jpg', 'png', 'tiff'])

    if "last_file" not in st.session_state:
        st.session_state.last_file = None
        st.session_state.doc_text = None

    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("⚠️ El archivo es demasiado grande. Sube un archivo de menos de 10 MB.")
            return

        if uploaded_file.name != st.session_state.last_file:
            # Procesar nuevo archivo
            st.session_state.last_file = uploaded_file.name
            file_extension = uploaded_file.name.split(".")[-1].lower()

            try:
                match file_extension:
                    case "pdf":
                        st.session_state.doc_text = process_pdf(uploaded_file)
                    case "csv":
                        st.session_state.doc_text = process_csv(uploaded_file)
                    case "xls" | "xlsx":
                        st.session_state.doc_text = process_excel(uploaded_file)
                    case "xml":
                        st.session_state.doc_text = process_xml(uploaded_file)
                    case "jpg" | "png" | "tiff":
                        st.session_state.doc_text = process_image(uploaded_file)
                    case _:
                        st.error("⚠️ Formato de archivo no compatible.")
                        return
            except Exception as e:
                st.error(f"⚠️ Error al procesar el archivo: {e}")
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