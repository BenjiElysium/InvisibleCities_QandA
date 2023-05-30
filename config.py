import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class SimpleConfig:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_region = os.getenv("PINECONE_ENV")
        self.replicate_api_key = os.getenv("REPLICATE_API_KEY")
