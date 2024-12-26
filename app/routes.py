from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from pymongo import MongoClient
from openai import OpenAI
from .tasks import generate_image_task
from .embedding_service import EmbeddingService
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Crear el Blueprint primero
routes = Blueprint("routes", __name__)

# Conectar a MongoDB y OpenAI
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Crear instancia del servicio de embeddings
embedding_service = EmbeddingService(mongo_client, openai_client)

db = mongo_client["image_generation"]
questions_collection = db["questions"]

def collection_exists(db, collection_name):
    return collection_name in db.list_collection_names()

@routes.route("/")
def home():
    images = list(questions_collection.find({"category": "inicio"}))
    return render_template("index.html", images=images)

@routes.route("/competition")
def competition():
    images = list(
        questions_collection.find({"category": "competition", "status": "completed"})
    )
    return render_template("competition.html", images=images)

@routes.route("/ranking")
def ranking():
    # Actualizar embeddings si es necesario
    embedding_service.update_prompt_embeddings()
    
    # Obtener submissions ordenadas por similitud
    ranked_submissions = embedding_service.get_ranked_submissions()
    
    return render_template("ranking.html", ranked_submissions=ranked_submissions)

@routes.route("/gallery")
def gallery():
    images = list(questions_collection.find({"category": "gallery"}))
    return render_template("gallery.html", images=images)

@routes.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        user_name = request.form.get("user_name")
        prompt = request.form.get("prompt")
        selected_image = request.form.get("selected_image", "familia.png")

        if not collection_exists(db, "questions"):
            questions_collection.create_index("user_name")

        question = {
            "user_name": user_name,
            "image_path": f"images/submit/{selected_image}",
            "prompt": prompt,
            "category": "competition",
            "status": "pending",
        }
        questions_collection.insert_one(question)

        generate_image_task.delay(prompt, user_name)
        flash("Tu solicitud ha sido enviada y la imagen se generará pronto.")
        return redirect(url_for("routes.competition"))

    # Obtener lista de imágenes disponibles
    submit_folder = os.path.join(current_app.static_folder, 'images', 'submit')
    available_images = [
        f for f in os.listdir(submit_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))
    ]
    
    return render_template("submit.html", available_images=available_images)
