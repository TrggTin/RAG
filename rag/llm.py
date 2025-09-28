import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


def get_gemini_embeddings(model: str = "models/embedding-001"):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable.")
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)


def get_gemini_chat(model: str = "gemini-2.5-flash", temperature: float = 0.1):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY environment variable.")
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=temperature, convert_system_message_to_human=True) 