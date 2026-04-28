import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    return GoogleGenerativeAIEmbeddings(model=model_name)
