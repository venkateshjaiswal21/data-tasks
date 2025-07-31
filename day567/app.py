import streamlit as st
from langchain.text_splitter import (
    CharacterTextSplitter,
    SpacyTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Chunking & FAISS Comparison", layout="wide")

HUGGINGFACE_MODEL = "all-MiniLM-L6-v2"

# -------------------- Helper Functions --------------------

def chunk_text(text, method, size=1000, overlap=50):
    if method == "Fixed-size":
        return CharacterTextSplitter(chunk_size=size, chunk_overlap=0).split_text(text)
    elif method == "Sentence-based":
        return SpacyTextSplitter().split_text(text)
    elif method == "Recursive":
        return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                             separators=["\n\n", "\n", ".", " ", ""]).split_text(text)
    return []

def embed_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL)
    return embeddings.embed_documents(chunks)

# -------------------- State Initialization --------------------
if "chunk_results" not in st.session_state:
    st.session_state["chunk_results"] = {}
if "file_text" not in st.session_state:
    st.session_state["file_text"] = ""

# -------------------- Day 6: Chunk + Embedding Comparison --------------------
st.title("📘 Day 5 + 6: Compare Chunking Methods with Embeddings")
uploaded_file = st.file_uploader("📎 Upload your .txt file", type=["txt"])
if uploaded_file:
    st.session_state["file_text"] = uploaded_file.read().decode("utf-8")

text_input = st.session_state["file_text"]

methods = ["Fixed-size", "Sentence-based", "Recursive"]
selected_methods = st.multiselect("✅ Select two chunking methods:", methods, default=methods[:2])

chunk_size = chunk_overlap = 0
if any(m in selected_methods for m in ["Fixed-size", "Recursive"]):
    chunk_size = st.number_input("📏 Chunk size:", 100, 5000, 1000, 100, key="chunk_size")
if "Recursive" in selected_methods:
    chunk_overlap = st.number_input("🔁 Overlap for Recursive:", 0, max(chunk_size-1, 0), 50, 10, key="chunk_overlap")

# Embedding button and logic
if st.button("🚀 Run Embedding"):
    st.session_state["chunk_results"] = {}
    if not text_input:
        st.warning("Upload text file.")
    elif len(selected_methods) != 2:
        st.warning("Select exactly 2 methods.")
    else:
        with st.spinner("Chunking and embedding..."):
            for method in selected_methods:
                # Safety for overlap
                overlap = chunk_overlap if method == "Recursive" else 0
                chunks = chunk_text(text_input, method, size=chunk_size, overlap=overlap)
                embeddings = embed_chunks(chunks)
                st.session_state["chunk_results"][method] = {
                    "chunks": chunks,
                    "embeddings": embeddings
                }
                st.success(f"✅ {len(chunks)} chunks embedded for '{method}'")
                st.markdown("**Sample Chunks:**")
                for i in range(min(3, len(chunks))):
                    st.code(chunks[i][:800] + ("..." if len(chunks[i]) > 800 else ""))

# -------------------- Day 7: FAISS Querying --------------------

st.title("📗 Day 7: FAISS Querying and Retrieval Comparison")

chunk_results = st.session_state["chunk_results"]

if chunk_results:
    st.markdown("### 🧠 Choose chunking method for querying")
    selected_query_method = st.selectbox("🔍 Select one method to run FAISS search on:", list(chunk_results.keys()))
    
    st.markdown("### ✍️ Enter your queries")
    query_input = st.text_area("Enter at least 5 queries (one per line)", height=150)
    queries = [q.strip() for q in query_input.strip().split("\n") if q.strip()]

    def run_faiss_search(chunks, queries):
        # Use already embedded chunks
        docs = [Document(page_content=c) for c in chunks]
        embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL)
        vectorstore = FAISS.from_documents(docs, embeddings)
        result = {}
        for q in queries:
            result[q] = vectorstore.similarity_search(q, k=3)
        return result

    if st.button("🔍 Run FAISS Querying"):
        if len(queries) < 5:
            st.warning("Please enter at least 5 queries.")
        else:
            chunks = chunk_results[selected_query_method]["chunks"]
            results = run_faiss_search(chunks, queries)
            st.subheader("📌 Query Results")
            for q in queries:
                st.markdown(f"#### 🔍 Query: `{q}`")
                docs = results[q]
                if docs:
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Top Match {i+1}:**")
                        st.code(doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else ""))
                else:
                    st.warning("No results found.")
else:
    st.info("Please complete chunking and embedding on Day 6 first.")
