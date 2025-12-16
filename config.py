import os
from dotenv import load_dotenv

load_dotenv()

LARAVEL_BASE_URL = os.getenv("LARAVEL_BASE_URL", "http://127.0.0.1:8000")
INTERNAL_TOKEN = os.getenv("INTERNAL_API_TOKEN", "")
MODEL_PATH = "models/model_latest.h5"

