from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import tempfile

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

from utils.rag import load_docs_from_paths, build_vectorstore, relevant_context
from utils.websearch import summarized_snippets

st.set_page_config(page_title="NeoStats AI Chatbot", page_icon="🤖", layout="wide")

SYSTEM_PROMPT = """You are a helpful AI assistant. If RAG context is provided between <context> tags, use it first.
If web results are provided between <web> tags, use them to ground your answer with brief citations (just the domain).
Write clearly and avoid making up facts.
"""

def instructions_page():
    st.title("NeoStats AI Engineer Use Case — Solution")
    st.markdown("""
**What this app demonstrates**  
- ✅ Retrieval-Augmented Generation (RAG) over uploaded PDFs/TXT/MD  
- ✅ Live Web Search fallback via Tavily  
- ✅ Response Modes: **Concise** vs **Detailed**  
- ✅ Modular code: config, models, utils  
""")

def chat_page():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    with st.sidebar:
        st.header("Controls")
        provider = st.selectbox("Model provider", ["groq","openai","gemini"])
        mode = st.radio("Response mode", ["Concise","Detailed"], horizontal=True)
        use_rag = st.toggle("Enable RAG (use uploaded docs)", value=True)
        use_web = st.toggle("Enable Web Search fallback", value=True)
        uploaded = st.file_uploader("Upload knowledge files", type=["pdf","txt","md"], accept_multiple_files=True)

        if uploaded:
            save_paths = []
            upload_dir = tempfile.gettempdir()  # system temp dir
            os.makedirs(upload_dir, exist_ok=True)
            for up in uploaded:
                path = os.path.join(upload_dir, up.name)
                with open(path, "wb") as f:
                    f.write(up.read())
                save_paths.append(path)
            docs = load_docs_from_paths(save_paths)
            st.session_state.vectorstore = build_vectorstore(docs)
            st.success(f"Indexed {len(docs)} documents.")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

    st.title("Chat")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Ask me anything...")
    if not user_input:
        return

    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build system + context
    system = SYSTEM_PROMPT
    ctx = ""
    if st.session_state.vectorstore and use_rag:
        ctx = relevant_context(st.session_state.vectorstore, user_input)
        if ctx:
            system += f"\n<context>\n{ctx}\n</context>"

    if mode == "Concise":
        system += "\nPrefer brief, to-the-point responses (3-5 sentences)."
    else:
        system += "\nProvide an in-depth, structured response when helpful."

    # Invoke model (Groq example)
    chat = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    messages = [SystemMessage(content=system)]
    for m in st.session_state.messages:
        role = m["role"]
        if role == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))

    try:
        resp = chat.invoke(messages).content
    except Exception as e:
        resp = f"(Model error: {e})."

    # Web fallback if response seems insufficient and enabled
    if use_web and (("I don't know" in resp) or len(resp) < 40):
        try:
            snippets = summarized_snippets(user_input, max_results=5)
        except Exception as e:
            snippets = ""
        if snippets:
            system_web = system + f"\n<web>\n{snippets}\n</web>\nUse the web snippets to answer and include brief source domains."
            try:
                resp = chat.invoke(
                    [SystemMessage(content=system_web), HumanMessage(content=user_input)]
                ).content
            except Exception as e:
                resp += f"\n(Web search error: {e})"

    st.session_state.messages.append({"role":"assistant","content":resp})
    with st.chat_message("assistant"):
        st.markdown(resp)

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Instructions","Chat"], index=1)
    if page == "Instructions":
        instructions_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
