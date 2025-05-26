# app.py (frontend)
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Detector de Emociones", layout="centered")
st.title("😊 Detector de Emociones Humanas a partir de Imágenes")

uploaded_file = st.file_uploader("Sube una imagen facial:", type=["jpg", "jpeg", "png"])

if st.button("Detectar emociones"):
    if uploaded_file is not None:
        try:
            with st.spinner("Analizando imagen..."):
                image_bytes = uploaded_file.read()
                response = requests.post(
                    "http://localhost:8000/predict",
                    files={"file": ("image.jpg", image_bytes, "image/jpeg")},
                )
                if response.status_code == 200:
                    results = response.json()
                    st.image(Image.open(io.BytesIO(image_bytes)), caption="Imagen subida", use_container_width=True)
                    st.subheader("Resultados:")
                    for item in results:
                        st.write(f"{item['label']}: {round(item['score'] * 100, 2)}%")
                        st.progress(min(int(item['score'] * 100), 100))
                else:
                    st.error("Error en la respuesta del servidor.")
        except Exception as e:
            st.error(f"Error al conectar con el backend: {e}")
    else:
        st.warning("Sube una imagen para analizar.")
