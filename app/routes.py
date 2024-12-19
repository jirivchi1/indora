from flask import Blueprint, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
from .tasks import generate_image_task
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Conectar a MongoDB
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["image_generation"]
questions_collection = db["questions"]

routes = Blueprint("routes", __name__)


def collection_exists(db, collection_name):
    return collection_name in db.list_collection_names()


@routes.route("/")
def home():
    images = list(questions_collection.find({"category": "inicio"}))
    return render_template("index.html", images=images)


@routes.route("/competition")
def competition():
    # Mostrar solo las imágenes completadas
    images = list(
        questions_collection.find({"category": "competition", "status": "completed"})
    )
    return render_template("competition.html", images=images)


@routes.route("/gallery")
def gallery():
    images = list(questions_collection.find({"category": "gallery"}))
    return render_template("gallery.html", images=images)


@routes.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        user_name = request.form.get("user_name")
        prompt = request.form.get("prompt")

        # Verificar si la colección existe (opcional)
        if not collection_exists(db, "questions"):
            # Crear índices u otras configuraciones iniciales si es necesario
            questions_collection.create_index("user_name")

        # Insertar en la base de datos con estado "pending"
        question = {
            "user_name": user_name,
            "image_path": "images/original/first.jpg",  # Ruta de la imagen original
            "prompt": prompt,
            "category": "competition",
            "status": "pending",
        }
        questions_collection.insert_one(question)

        # Iniciar la tarea de Celery para generar la imagen
        generate_image_task.delay(prompt, user_name)

        # Flash message para notificar al usuario
        flash("Tu solicitud ha sido enviada y la imagen se generará pronto.")

        # Redirigir a /competition para ver el nuevo dato (pendiente no se muestra)
        return redirect(url_for("routes.competition"))

    # Si es GET, mostramos el formulario
    return render_template("submit.html")
