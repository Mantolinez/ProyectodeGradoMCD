📄Detección Automática de Requisitos Contractuales en Minutas
Este proyecto implementa una aplicación en Streamlit que permite cargar minutas en formato PDF, aplicar preprocesamiento de texto, vectorización (TF-IDF o Word2Vec) y clasificación automática de requisitos contractuales utilizando modelos entrenados previamente.

🚀Objetivo
Automatizar la detección de requisitos contractuales en documentos legales mediante modelos de aprendizaje automático, para asistir en procesos de revisión y control documental.

🧠 Requisitos Identificados
Los modelos están entrenados para identificar los siguientes requisitos:

TF-IDF + SVC:
Retención en garantía
Gastos reembolsables
Cláusula de Cesión
Socialización
Subcontratación
Uso de opción
Reajuste salarial
GAB-F-213
GAB-F-214
GAB-F-221
Reunión de inicio

Word2Vec + SVC:
GAB-F-105
Garantías y seguros
Reajuste de tarifas y precios

🗂️ Estructura del Proyecto
ProyectodeGradoMCD/

app.py → Código principal de la app Streamlit
modelos_finales/ → Carpeta con modelos y vectorizadores
modelo_retencion_en_garantia.pkl
modelo_garantias_y_seguros.pkl
..
modelo_word2vec.model
vectorizador_tfidf.pkl
README.md → Este archivo
.gitattributes → Configuración para Git LFS

📦 Requisitos de Instalación
Para ejecutar el proyecto localmente, instala las siguientes dependencias:

streamlit
scikit-learn
joblib
gensim
nltk
unidecode
PyPDF2

▶️ Cómo ejecutar la aplicación
Ejecuta este comando desde la raíz del proyecto:

bash
Copiar
Editar
streamlit run app.py
Esto abrirá una interfaz web donde podrás cargar una minuta en formato PDF y ver automáticamente los requisitos identificados.

🧠 Notas Técnicas
Cada archivo .pkl corresponde a un modelo entrenado con SVC para una etiqueta específica.
El archivo vectorizador_tfidf.pkl contiene el TF-IDF global.
modelo_word2vec.model es el modelo Word2Vec entrenado previamente.

El texto extraído del PDF es preprocesado: minúsculas, limpieza, stopwords y stemming.
Los vectores resultantes se pasan al modelo correspondiente, y se muestra un resumen con los resultados.

👨‍💻 Desarrollado por
Mónica Jazmín Antolínez Becerra
Adriana María Guiza Saavedra
Maestría en Ciencia de Datos – Proyecto de Grado
Potificia Unviersidad Javeriana Cali
