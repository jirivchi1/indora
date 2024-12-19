from app import create_app  # Importa la función create_app de app/__init__.py

# Ejecuta la aplicación
if __name__ == "__main__":
    app = (
        create_app()
    )  # Crea una instancia de la aplicación usando la función create_app
    app.run(debug=True)
