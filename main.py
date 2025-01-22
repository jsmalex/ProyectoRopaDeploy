import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from fastapi import FastAPI, File, UploadFile
from sklearn.metrics.pairwise import cosine_similarity
import io
import json
from typing import List



# Cargar el modelo para generar embeddings
model_without_top = tf.keras.models.load_model('Modelo_final_embenddings_ropa.keras')

# Cargar la información del JSON con embeddings y rutas
with open('dataset_info.json', 'r') as f:
    dataset_info = json.load(f)

# Extraer los embeddings y las IDs del JSON
embeddings_ropa = np.array([item["embedding"] for item in dataset_info])
image_ids = [item["id"] for item in dataset_info]

# Crear la aplicación FastAPI
app = FastAPI()

# Función para obtener el embedding de una imagen
def get_image_embedding(img_bytes: bytes, model):
    img = image.load_img(io.BytesIO(img_bytes), target_size=(299, 299))  # Cargar la imagen desde bytes
    img_array = image.img_to_array(img)  # Convertir a array numpy
    img_array = np.expand_dims(img_array, axis=0)  # Añadir batch dimension
    img_array = img_array / 255.0  # Normalizar
    embedding = model.predict(img_array)  # Obtener el embedding
    return embedding

# Ruta de la API para recibir la imagen y devolver los 5 IDs más similares
@app.post("/compare-image/")
async def compare_image(file: UploadFile = File(...)):
    # Leer la imagen subida
    img_bytes = await file.read()

    # Obtener el embedding de la imagen
    image_embedding = get_image_embedding(img_bytes, model_without_top)

    # Calcular la similitud coseno entre el embedding de la imagen y los embeddings guardados
    similarities = cosine_similarity(image_embedding, embeddings_ropa)

    # Obtener las 5 imágenes más similares (con mayor similitud)
    top_5_similar_indices = np.argsort(similarities[0])[::-1][:5]

    # Obtener las IDs correspondientes a las imágenes más similares
    similar_ids = [image_ids[idx] for idx in top_5_similar_indices]

    # Devolver las IDs como respuesta
    return {"similar_ids": similar_ids}
