from typing import Literal
from config.config import (
    DEFAULT_CHAT_PROVIDER, GROQ_MODEL, OPENAI_MODEL, GEMINI_MODEL,
    OPENAI_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY
)

def get_chat_model(provider: Literal["groq","openai","gemini"] = None):
    provider = provider or DEFAULT_CHAT_PROVIDER
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=GROQ_MODEL, groq_api_key=GROQ_API_KEY)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL,
                                      google_api_key=GOOGLE_API_KEY)
    raise ValueError(f"Unknown provider: {provider}")
