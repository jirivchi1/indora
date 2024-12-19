from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Connect to MongoDB
mongo_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongo_client["image_generation"]

# Access collections
questions_collection = db["questions"]  # Updated collection name

# Initialize the questions collection if not already done
questions_data = [
    {
        "user_name": "jose",
        "image_path": "images/original/first.png",
        "prompt": "na manzana a la izquierda, dos mandarinas (una completa, una con gajos) en el centro y aguacate a la derecha cortado por la mitad.",
        "category": "inicio",
        "status": "completed",
        "image_filename": "first.png",
    },
]

for question in questions_data:
    existing_question = db.questions.find_one({"question_id": question["question_id"]})
    if not existing_question:
        db.questions.insert_one(question)
