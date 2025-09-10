from typing import List, Optional
import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from models.embeddings import get_embedding_model
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

def load_docs_from_paths(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(p)
            docs.extend(loader.load())
        elif ext in {".txt", ".md"}:
            docs.extend(TextLoader(p, encoding="utf-8").load())
        else:
            # Fallback: try text loader
            try:
                docs.extend(TextLoader(p, encoding="utf-8").load())
            except Exception:
                pass
    return docs

def build_vectorstore(docs: List[Document]):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    embeddings = get_embedding_model()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def relevant_context(vs, query: str, k: int = TOP_K) -> str:
    if vs is None:
        return ""
    results = vs.similarity_search(query, k=k)
    joined = "\n\n".join([f"[{i+1}] " + r.page_content for i, r in enumerate(results)])
    return joined
