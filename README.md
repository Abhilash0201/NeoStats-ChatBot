# NeoStats AI Chatbot — Reference Solution

This app demonstrates:
- **RAG** over uploaded PDFs/TXT/MD (FAISS + OpenAI/Gemini embeddings)
- **Live Web Search** fallback (Tavily)
- **Concise vs Detailed** response modes
- Modular codebase: `config/`, `models/`, `utils/`, `app.py`

## Quickstart

1. **Python 3.10+** recommended.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables for providers you intend to use:
   ```bash
   export OPENAI_API_KEY=...
   export GROQ_API_KEY=...
   export GOOGLE_API_KEY=...
   export TAVILY_API_KEY=...
   ```
4. Run locally:
   ```bash
   streamlit run app.py
   ```

## Features

- **RAG**: Upload documents in the sidebar; they are chunked and indexed in FAISS. Queries retrieve top-k chunks to ground responses.
- **Web Search**: If the first model response is uncertain/short, the app queries Tavily and re-answers using summarized snippets.
- **Modes**: Toggle **Concise** or **Detailed** to control verbosity.
- **Providers**: Choose **Groq**, **OpenAI**, or **Gemini** from the sidebar.

## Deploy to Streamlit Cloud

1. Push the folder to a GitHub repository.
2. On https://streamlit.io/cloud, create a new app pointing to `app.py`.
3. Add the following **secrets** in Streamlit Cloud:
   ```toml
   OPENAI_API_KEY="..."
   GROQ_API_KEY="..."
   GOOGLE_API_KEY="..."
   TAVILY_API_KEY="..."
   DEFAULT_CHAT_PROVIDER="groq"
   GROQ_MODEL="llama-3.1-70b-versatile"
   OPENAI_MODEL="gpt-4o-mini"
   GEMINI_MODEL="gemini-1.5-pro"
   EMBEDDING_PROVIDER="openai"
   OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
   GEMINI_EMBEDDING_MODEL="text-embedding-004"
   CHUNK_SIZE="1200"
   CHUNK_OVERLAP="150"
   TOP_K="4"
   ```
4. Deploy.

## File Map

```
AI_UseCase/
├─ app.py
├─ requirements.txt
├─ README.md
├─ config/
│  └─ config.py
├─ models/
│  ├─ llm.py
│  └─ embeddings.py
└─ utils/
   ├─ rag.py
   └─ websearch.py
```

---

**Notes**:  
- You can extend document loaders (CSV, DOCX) by adding loaders in `utils/rag.py`.  
- If you prefer SerpAPI/Bing instead of Tavily, replace `utils/websearch.py`.  
