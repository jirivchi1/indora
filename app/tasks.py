from .celery_app import celery
from flask import current_app
from datetime import datetime
import os
import requests
from pymongo import MongoClient
from openai import OpenAI
from .embedding_service import EmbeddingService

@celery.task()
def generate_image_task(prompt, username):
    username = username.strip()

    mongodb_uri = current_app.config["MONGODB_URI"]
    openai_api_key = current_app.config["OPENAI_API_KEY"]

    mongo_client = MongoClient(mongodb_uri)
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Crear instancia del servicio de embeddings
    embedding_service = EmbeddingService(mongo_client, openai_client)

    db = mongo_client["image_generation"]
    questions_collection = db["questions"]

    try:
        # Generar imagen
        response = openai_client.images.generate(
            prompt=prompt, n=1, size="1024x1024", response_format="url"
        )
    except Exception as e:
        questions_collection.update_one(
            {"user_name": username, "prompt": prompt},
            {"$set": {"status": "failed", "error": str(e)}},
        )
        return None

    image_url = response.data[0].url

    try:
        image_data = requests.get(image_url).content
    except Exception as e:
        questions_collection.update_one(
            {"user_name": username, "prompt": prompt},
            {"$set": {"status": "failed", "error": "Failed to download image"}},
        )
        return None

    image_filename = f"{username}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    image_path = os.path.join(
        current_app.root_path, "static", "images", "competition", image_filename
    )
    with open(image_path, "wb") as file:
        file.write(image_data)

    # Generar embedding para el prompt
    prompt_embedding = embedding_service.get_embedding(prompt)

    # Actualizar documento con imagen y embedding
    questions_collection.update_one(
        {"user_name": username, "prompt": prompt},
        {
            "$set": {
                "image_filename": image_filename,
                "status": "completed",
                "prompt_embedding": prompt_embedding
            }
        },
    )

    return image_filename
