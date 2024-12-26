from openai import OpenAI
import numpy as np
from pymongo import MongoClient

class EmbeddingService:
    def __init__(self, mongo_client, openai_client):
        self.mongo_client = mongo_client
        self.openai_client = openai_client
        self.db = self.mongo_client["image_generation"]
        self.questions_collection = self.db["questions"]

    def get_embedding(self, text):
        """Genera el embedding vectorial para un texto dado."""
        response = self.openai_client.embeddings.create(
            input=[text], model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return embedding

    def update_prompt_embeddings(self):
        """
        Genera embeddings para los prompts de las categorías 'inicio' y 'competition'
        que aún no los tienen.
        """
        for doc in self.questions_collection.find(
            {
                "prompt_embedding": {"$exists": False},
                "category": {"$in": ["inicio", "competition"]},
            }
        ):
            prompt = doc.get("prompt")
            if prompt:
                prompt_embedding = self.get_embedding(prompt)
                self.questions_collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"prompt_embedding": prompt_embedding}},
                )

    def cosine_similarity(self, a, b):
        """Calcula la similitud coseno entre dos vectores."""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get_ranked_submissions(self):
        """Obtiene las submissions ordenadas por similitud."""
        inicio_doc = self.questions_collection.find_one({"category": "inicio"})
        if not inicio_doc:
            return []

        # Asegurarse de que el documento inicio tiene embedding
        if not inicio_doc.get("prompt_embedding"):
            inicio_embedding = self.get_embedding(inicio_doc["prompt"])
            self.questions_collection.update_one(
                {"_id": inicio_doc["_id"]},
                {"$set": {"prompt_embedding": inicio_embedding}},
            )
            inicio_doc["prompt_embedding"] = inicio_embedding

        # Obtener submissions completadas
        submissions = list(
            self.questions_collection.find(
                {"category": "competition", "status": "completed"}
            )
        )

        # Calcular similitudes
        ranked_submissions = []
        for doc in submissions:
            if not doc.get("prompt_embedding"):
                prompt_embedding = self.get_embedding(doc["prompt"])
                self.questions_collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"prompt_embedding": prompt_embedding}},
                )
                doc["prompt_embedding"] = prompt_embedding

            similarity = self.cosine_similarity(
                inicio_doc["prompt_embedding"], doc["prompt_embedding"]
            )
            doc["similarity"] = round(similarity * 100, 2)  # Convert to percentage
            ranked_submissions.append(doc)

        # Ordenar por similitud descendente
        ranked_submissions.sort(key=lambda x: x["similarity"], reverse=True)
        return ranked_submissions
