# 📄 Chat con PDFs usando LLMs

Este proyecto es una aplicación de chat que permite interactuar con documentos PDF utilizando modelos de lenguaje grandes (LLMs). Está construida con **Streamlit**, **LangChain** y **OpenAI**.

## 🚀 Instalación y Configuración

### 1️⃣ Clonar el repositorio
```sh
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
```

### 2️⃣ Activar el entorno virtual
🔹 En macOS/Linux:
```sh
source env/bin/activate
```
🔹 En Windows:

```sh
env\Scripts\activate
```

### 3️⃣ Instalar las dependencias

```sh
pip install -r requirements.txt
```

### 🏃 Ejecución del proyecto
Dentro de la carpeta env, corre el siguiente comando en la terminal:
```sh
streamlit run app.py
```

### 📝 ¿Cómo funciona?
Sube un archivo PDF desde la interfaz de Streamlit.
La aplicación procesará el PDF y generará embeddings con FAISS.
Puedes hacer preguntas sobre el contenido del PDF en la barra de entrada.
La respuesta se generará utilizando OpenAI GPT-4o.

### ⚙️ Actualización 

Se actualizo para que pueda procesar pdf, csv, xml, xlsx y xls