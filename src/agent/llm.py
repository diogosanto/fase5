import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def get_llm() -> ChatGoogleGenerativeAI:
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
    )
