import os
import numpy as np
from openai import OpenAI
from pymongo import MongoClient
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar el cliente de OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Conectar a MongoDB
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["image_generation"]

# Acceder a la colección
questions_collection = db["questions"]


def get_embedding(text):
    """Genera el embedding vectorial para un texto dado."""
    response = openai_client.embeddings.create(
        input=[text], model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return embedding


def update_prompt_embeddings():
    """
    Genera embeddings para los prompts de las categorías 'inicio' y 'competition'
    que aún no los tienen. Excluye la categoría 'gallery'.
    """
    # Filtrar documentos que no tienen embedding y que pertenecen a 'inicio' o 'competition'
    for doc in questions_collection.find(
        {
            "prompt_embedding": {"$exists": False},
            "category": {"$in": ["inicio", "competition"]},
        }
    ):
        prompt = doc.get("prompt")
        if prompt:
            prompt_embedding = get_embedding(prompt)
            questions_collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"prompt_embedding": prompt_embedding}},
            )
            print(
                f"Actualizado embedding para el prompt de {doc['user_name']} en la categoría '{doc['category']}'."
            )


def cosine_similarity(a, b):
    """Calcula la similitud coseno entre dos vectores."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank_competitions():
    """Ranquea los prompts de la categoría 'competition' según su similitud con el prompt de 'inicio'."""
    # Obtener el documento de la categoría 'inicio'
    inicio_doc = questions_collection.find_one({"category": "inicio"})
    if not inicio_doc:
        print("No se encontró un documento con la categoría 'inicio'.")
        return

    inicio_embedding = inicio_doc.get("prompt_embedding")
    if not inicio_embedding:
        print("Generando embedding para el prompt de 'inicio'.")
        inicio_embedding = get_embedding(inicio_doc["prompt"])
        questions_collection.update_one(
            {"_id": inicio_doc["_id"]},
            {"$set": {"prompt_embedding": inicio_embedding}},
        )
        print("Embedding generado y actualizado para 'inicio'.")

    # Obtener todos los documentos de la categoría 'competition'
    competition_docs = list(questions_collection.find({"category": "competition"}))
    if not competition_docs:
        print("No se encontraron documentos con la categoría 'competition'.")
        return

    # Asegurarse de que todos los documentos de 'competition' tengan embeddings
    for doc in competition_docs:
        if not doc.get("prompt_embedding"):
            print(
                f"Generando embedding para el prompt de {doc['user_name']} en 'competition'."
            )
            prompt_embedding = get_embedding(doc["prompt"])
            questions_collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"prompt_embedding": prompt_embedding}},
            )
            print(f"Embedding generado y actualizado para {doc['user_name']}.")

    # Calcular similitudes
    similarities = []
    for doc in competition_docs:
        comp_embedding = doc.get("prompt_embedding")
        if comp_embedding:
            sim = cosine_similarity(inicio_embedding, comp_embedding)
            similarities.append(
                {
                    "user_name": doc.get("user_name"),
                    "prompt": doc.get("prompt"),
                    "similarity": sim,
                    "image_path": doc.get("image_path"),
                }
            )

    # Ordenar por similitud descendente
    similarities.sort(key=lambda x: x["similarity"], reverse=True)

    # Imprimir el ranking
    print("\nRanking de similitud de 'competition' respecto a 'inicio':")
    for idx, item in enumerate(similarities, start=1):
        print(
            f"{idx}. Usuario: {item['user_name']}, Similitud: {item['similarity']:.4f}"
        )
        print(f"   Prompt: {item['prompt']}")
        print(f"   Imagen: {item['image_path']}\n")


if __name__ == "__main__":
    print("Actualizando embeddings faltantes...")
    update_prompt_embeddings()
    print("Ranqueando competiciones...")
    rank_competitions()
