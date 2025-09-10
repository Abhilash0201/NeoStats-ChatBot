from typing import Literal
from config.config import (
    EMBEDDING_PROVIDER, OPENAI_EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL,
    OPENAI_API_KEY, GOOGLE_API_KEY
)

def get_embedding_model(provider: Literal["openai", "groq","gemini"] = None):
    provider = provider or EMBEDDING_PROVIDER
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL,
                                api_key=OPENAI_API_KEY)
    elif provider == "groq":
        from langchain.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")    
    
    elif provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL,
                                            google_api_key=GOOGLE_API_KEY)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
