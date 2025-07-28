import streamlit as st
from langchain.text_splitter import (
    CharacterTextSplitter,
    SpacyTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# -------------------- UI: Intro --------------------
st.title("📚 Day 5: RAG + Semantic Chunk Search with FAISS")

with st.expander("ℹ️ About RAG and FAISS"):
    st.markdown("""
    - **RAG**: Combines retrieval + generation for smart answers.
    - **FAISS**: Fast vector database for semantic search.
    - **HuggingFace Embeddings**: Turns text into numerical vectors for similarity.

    This app:
    - Splits your text into chunks.
    - Embeds them.
    - Uses FAISS to find top matches for each query.
    """)

# -------------------- Upload Text File --------------------
uploaded_file = st.file_uploader("📎 Upload your .txt document", type=["txt"])
text_input = uploaded_file.read().decode("utf-8") if uploaded_file else ""

# -------------------- User Queries --------------------
st.subheader("❓ Your Queries")
query_input = st.text_area("Enter at least 5 queries (one per line)", height=150)
queries = [q.strip() for q in query_input.strip().split("\n") if q.strip()]

# -------------------- Chunking Config --------------------
st.subheader("⚙️ Chunking Settings")
chunking_method = st.selectbox("Select a chunking strategy:", ("Fixed-size", "Sentence-based", "Recursive"))

chunk_size = chunk_overlap = 0
if chunking_method in ["Fixed-size", "Recursive"]:
    chunk_size = st.number_input("📏 Chunk size:", min_value=100, max_value=10000, value=1000, step=100)
    if chunking_method == "Recursive":
        chunk_overlap = st.number_input("🔁 Chunk overlap:", min_value=0, max_value=chunk_size - 1, value=50, step=10)

# -------------------- Chunking Function --------------------
def chunk_text(text, method, size=None, overlap=0):
    if method == "Fixed-size":
        splitter = CharacterTextSplitter(chunk_size=size, chunk_overlap=0)
    elif method == "Sentence-based":
        splitter = SpacyTextSplitter()
    elif method == "Recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
    else:
        return []
    return splitter.create_documents([text])

# -------------------- Search with FAISS --------------------
def search_with_faiss(chunks, queries):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    results = {}
    for query in queries:
        docs = vectorstore.similarity_search(query, k=3)
        results[query] = docs
    return results

# -------------------- Main Action --------------------
if st.button("🚀 Chunk and Search"):
    if not text_input.strip():
        st.warning("Please upload a text file.")
    elif len(queries) < 5:
        st.warning("Please enter at least 5 queries.")
    else:
        # Step 1: Chunk
        st.info("🔪 Chunking text...")
        docs = chunk_text(text_input, chunking_method, size=chunk_size, overlap=chunk_overlap)
        st.success(f"✅ {len(docs)} chunks created.")

        # Step 2: Semantic search using FAISS
        st.info("🔍 Running semantic search...")
        faiss_results = search_with_faiss(docs, queries)

        # Step 3: Display matches
        st.subheader("📌 Top Matching Chunks (Semantic Search)")
        for query in queries:
            st.markdown(f"### 🔍 Query: `{query}`")
            results = faiss_results[query]
            if results:
                for i, doc in enumerate(results):
                    st.markdown(f"**Top Match {i+1}:**")
                    st.code(doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else ""))
            else:
                st.warning("No matching chunks found.")

# -------------------- References --------------------
st.markdown("---")
st.markdown("### 📚 References")
st.markdown("- [RAG Overview (SingleStore)](https://www.singlestore.com/blog/a-guide-to-retrieval-augmented-generation-rag/)")
st.markdown("- [Chunking Techniques (F22 Labs)](https://www.f22labs.com/blogs/7-chunking-strategies-in-rag-you-need-to-know/)")
