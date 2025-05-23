import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_PATH = PROCESSED_DATA_DIR / "faiss_index"

# Model settings
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.0-flash"

# Document processing settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Search settings
TOP_K_RESULTS = 3
