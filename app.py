import streamlit as st # es una libreria para crear aplicaciones web visuales de datos rapido
import joblib # es para cargar nuestro modelo congelado
import numpy as np          # Sirve para hacer operaciones matemáticas avanzadas y manejar matrices o vectores numéricos.

# ponemos un titulo a nuestra web
st.title("🌍 Predicciones Globales: Esperanza de Vida")
st.write("Este modelo predice los años de vida promedio basados en la educación del país.")

# cargar el modelo que he entrenado en colab (asegurar que todo esten la misma carpeta en el portatil)
modelo = joblib.load('modelo_lineal.pkl')

#creamos una barra lateral para que el usuario pueda introducir los datos
# min_value=0, max_value=22, value=12 (valor por defecto), step=1
años_escuela = st.slider("Selecciona los años promedio de escolaridad del país:", 0, 22, 12, 1)

# Cuando el usuario mueva la barra, deberia de hacer la predicción en tiempo real
# El modelo necesita los datos en formato matriz [[valor]]
entrada = np.array([[años_escuela]])
prediccion_vida = modelo.predict(entrada)[0]

# 5. Mostramos el resultado final en la pantalla con un diseño llamativo
st.success(f"La esperanza de vida estimada para este país es de: **{prediccion_vida:.1f} años**")
