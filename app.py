# -*- coding: utf-8 -*-
import streamlit as st
import joblib
from gensim.models import Word2Vec
import numpy as np
import PyPDF2
import os
from nltk.tokenize import word_tokenize
from unidecode import unidecode
from nltk.stem import SnowballStemmer


# Configuración
st.set_page_config(page_title="Requisitos Contractuales", layout="wide")
st.title("📄 Identificación Automática de Requisitos Contractuales en Minutas")

# Cargar vectorizadores
vectorizador_tfidf = joblib.load("modelos_finales/vectorizador_tfidf.pkl")
modelo_w2v = Word2Vec.load("modelos_finales/modelo_word2vec.model")

# Mapas de modelos por etiqueta
etiquetas_tfidf = ["Retención en garantía", "Gastos reembolsables", "Cláusula de Cesión", "Socialización", "Subcontratación","Uso de opción", "Reajuste salarial",  "GAB-F-213", "GAB-F-214", "GAB-F-221", "Reunión de inicio"]
etiquetas_w2v = ["GAB-F-105","Garantías y seguros","Reajuste de tarifas y precios" ]

# Funciones auxiliares
def extraer_texto_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    texto = ""
    for page in reader.pages:
        texto += page.extract_text() + "\n"
    return texto


stemmer = SnowballStemmer('spanish')

def preprocesar_texto(texto):
    texto = texto.lower()
    texto = unidecode(texto)
    texto = "".join([c for c in texto if c.isalnum() or c.isspace()])
    tokens = word_tokenize(texto)
    tokens_stemmed = [stemmer.stem(token) for token in tokens]
    texto = " ".join(tokens_stemmed)
    return texto

def vector_promedio(doc, modelo, dimension=100):
    tokens = word_tokenize(doc)
    vectores = [modelo.wv[word] for word in tokens if word in modelo.wv]
    return np.mean(vectores, axis=0) if vectores else np.zeros(dimension)

# Subida de archivo
archivo = st.file_uploader("Adjunta una minuta en formato PDF", type=["pdf"])

if archivo:
    texto_original = extraer_texto_pdf(archivo)
    texto_proc = preprocesar_texto(texto_original)

    st.subheader("Texto Preprocesado")
    st.text_area("Contenido", texto_proc, height=200)

    resultados = {}
  
    # Clasificación con TF-IDF
    X_tfidf = vectorizador_tfidf.transform([texto_proc]).toarray()
    for etiqueta in etiquetas_tfidf:
        nombre_archivo = f"modelo_{unidecode(etiqueta.lower().replace(' ', '_'))}.pkl"
        modelo = joblib.load(os.path.join("modelos_finales", nombre_archivo))
        pred = modelo.predict(X_tfidf)[0]
        
        # Mostrar predicción
        resultados[etiqueta] = "✅ Requisito Identificado" if pred == 1 else "❌ Requisito No Identificado"

    

    # Clasificación con Word2Vec
    X_w2v = vector_promedio(texto_proc, modelo_w2v, dimension=100).reshape(1, -1)
    for etiqueta in etiquetas_w2v:
        nombre_archivo = f"modelo_{unidecode(etiqueta.lower().replace(' ', '_'))}.pkl"
        modelo = joblib.load(os.path.join("modelos_finales", nombre_archivo))
        pred = modelo.predict(X_w2v)[0]

        resultados[etiqueta] = "✅ Requisito Identificado" if pred == 1 else "❌ Requisito No Identificado"


    import pandas as pd

    # Mostrar resultados 
    st.subheader("📋 Identificación de Requisitos")

    # Construir DataFrame estructurado
    df_resultados = pd.DataFrame([
       {
        "Requisito": etiqueta,
        "¿Identificado?": "Sí" if "✅" in resultado else "No",
        "Resultado": resultado
       }
      for etiqueta, resultado in resultados.items()
    ])

    # Mostrar tabla
    st.dataframe(df_resultados, use_container_width=True)
