import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from PyPDF2 import PdfReader


# CORRECTION 1: Improved chunking function for more robust document splitting
def chunk_text(text, chunk_size=512, chunk_overlap=50):
    """
    Splits text into overlapping chunks. First by paragraphs, then by size.
    """
    if not isinstance(text, str) or not text.strip():
        return []
        
    # Split by paragraphs first to respect document structure
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    all_chunks = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            all_chunks.append(para)
        else:
            # If a paragraph is too long, split it into smaller chunks
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunk = para[start:end]
                all_chunks.append(chunk)
                start += chunk_size - chunk_overlap
                
    return [chunk for chunk in all_chunks if chunk]


# 1. Streamlit UI: Sidebar and Inputs
st.set_page_config(page_title="Conference Proceedings Summarizer", layout="wide")
st.title("Conference Proceedings Summarizer using (RAG) 🚀")

# Initialize session state variables
if 'answer' not in st.session_state:
    st.session_state.answer = ""
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

# Document Upload
st.sidebar.title("Step 1: Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload text files (.txt)", type="txt", accept_multiple_files=True
)
uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDFs (.pdf)", type="pdf", accept_multiple_files=True
)

raw_docs = []
if uploaded_pdfs:
    for pdf_file in uploaded_pdfs:
        try:
            reader = PdfReader(pdf_file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            raw_docs.append(text)
        except Exception as e:
            st.sidebar.error(f"Error reading {pdf_file.name}: {e}")

if uploaded_files:
    for file in uploaded_files:
        raw_docs.append(file.read().decode("utf-8"))

# Split documents into chunks for fine-grained indexing
docs = []
with st.spinner("Chunking documents..."):
    for i, doc in enumerate(raw_docs):
        # Using the new, more robust chunking function
        docs.extend(chunk_text(doc))

if not docs:
    st.warning("Please upload at least one document to begin.")
    st.stop()

st.sidebar.success(f"Processed {len(raw_docs)} documents into {len(docs)} chunks.")

# 2. Embedder and Qdrant Setup (cache for speed)
@st.cache_resource
def get_resources(_docs): # Pass docs to ensure cache invalidation on new docs
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(_docs, show_progress_bar=True, normalize_embeddings=True)
    
    client = QdrantClient(":memory:")
    client.recreate_collection(
        collection_name="docs",
        vectors_config=qm.VectorParams(size=384, distance=qm.Distance.COSINE),
    )
    
    client.upsert(
        collection_name="docs",
        points=[
            qm.PointStruct(id=i, vector=embeddings[i].tolist(), payload={"text": _docs[i]})
            for i in range(len(_docs))
        ],
        wait=True
    )
    return embedder, client

embedder, client = get_resources(docs)

# CORRECTION 2: Upgraded to a more powerful generator model
@st.cache_resource
def get_generator():
    """Load and cache the text generation model."""
    model_name = "google/flan-t5-base" # Upgraded from 'small' to 'base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

generator = get_generator()


# 4. Query Input and Retrieval
st.header("Step 2: Ask a Question")
query = st.text_input("Your Question:", value="What is the main idea of the document?")
top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=10, value=3)

# CORRECTION 5: Clear previous answer if query changes
if query != st.session_state.last_query:
    st.session_state.answer = ""
    st.session_state.last_query = query

def retrieve(query, top_k):
    """Retrieve top_k relevant text chunks from Qdrant."""
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    response = client.search(collection_name="docs", query_vector=q_emb, limit=top_k)
    return [hit.payload["text"] for hit in response]

# Main layout columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Retrieved Context")
    contexts = retrieve(query, top_k)
    for i, ctx in enumerate(contexts):
        st.markdown(f"**Chunk {i+1}:**")
        st.info(ctx)

with col2:
    st.subheader("Generated Answer")
    # CORRECTION 3: Improved prompt engineering
    prompt_template = """
    Answer the question based *only* on the following context. If the context does not contain the answer, say "The context does not contain the answer to this question."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    context_str = "\n\n".join(contexts)
    prompt = prompt_template.format(context=context_str, question=query)

    if st.button("Generate Answer"):
        with st.spinner("🧠 Generating answer..."):
            # CORRECTION 4: Added min_length to encourage more complete answers
            generated = generator(
                prompt, 
                max_length=1024, 
                min_length=50, # Encourage more detailed answers
                clean_up_tokenization_spaces=True
            )[0]["generated_text"]
            st.session_state.answer = generated

    # CORRECTION 5: Simplified and corrected answer display logic
    if st.session_state.answer:
        st.success(st.session_state.answer)

# 6. Evaluation 
with st.expander("Step 3: Evaluate Answer (Optional)"):
    ground_truth = st.text_area("Paste the ground-truth answer here:")
    if st.button("Evaluate") and ground_truth:
        if not st.session_state.answer:
            st.warning("Please generate an answer first before evaluation.")
        else:
            pred_tokens = st.session_state.answer.lower().split()
            ref_tokens = ground_truth.lower().split()
            common_tokens = set(pred_tokens) & set(ref_tokens)
            
            if not pred_tokens or not ref_tokens:
                st.write("Cannot evaluate with empty prediction or reference.")
            else:
                precision = len(common_tokens) / len(pred_tokens)
                recall = len(common_tokens) / len(ref_tokens)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                exact_match = 1 if " ".join(pred_tokens) == " ".join(ref_tokens) else 0

                st.write(f"**F1-Score:** `{f1:.3f}`")
                st.write(f"**Precision:** `{precision:.3f}`")
                st.write(f"**Recall:** `{recall:.3f}`")
                st.write(f"**Exact Match:** `{exact_match}`")