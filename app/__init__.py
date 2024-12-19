from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()


def create_app():
    app = Flask(__name__)

    # Configuración de Flask
    app.config.update(
        CELERY_BROKER_URL=os.getenv("CELERY_BROKER_URL"),
        CELERY_RESULT_BACKEND=os.getenv("CELERY_RESULT_BACKEND"),
        MONGODB_URI=os.getenv("MONGODB_URI"),
        OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
    )

    # Establecer la secret_key para manejar sesiones y mensajes flash
    app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")


    # Importar y registrar Blueprints
    from .routes import routes

    app.register_blueprint(routes)

    # Importar celery y actualizar su configuración
    from .celery_app import celery

    celery.conf.update(app.config)

    # Configurar la tarea con contexto de aplicación
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask

    # Exponer celery en el contexto de la aplicación
    app.celery = celery

    return app


app = create_app()
