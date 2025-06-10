from dataclasses import dataclass
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import faiss
import pickle
import requests
import logging
import time
import mlflow
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


import nltk
import os
import nltk

# Check if punkt already exists before downloading
nltk_data_path = nltk.data.path[0]
if not os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
    nltk.download('punkt')
# Configuration
@dataclass
class Config:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    llm_model: str = "llama3.2:1b-instruct-q3_K_S"
    ollama_url: str = "http://localhost:11434/api/chat"
    faiss_index_path: str = "index.faiss"
    chunks_path: str = "chunks.pkl"
    retrieve_k: int = 10
    top_k: int = 5
    max_context_chars: int = 12000
    llm_timeout: int = 240
    temperature: float = 0.7
    num_follow_up_questions: int = 3
    comparison_model= str = "llama3.2:1b"  # bigger Model for answer comparison

CONFIG = Config()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Load models and data at startup
# Define cache directory for models
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
os.environ["HF_HOME"] = "./model_cache"

# Initialize models with cache_folder parameter
logger.info("Starting application...")
logger.info("Loading embedding model...")
embedder = SentenceTransformer(CONFIG.embedding_model, cache_folder="./model_cache")
logger.info("Loading reranker model...")
reranker = CrossEncoder(CONFIG.reranker_model, cache_folder="./model_cache")
logger.info("Loading FAISS index...")
index = faiss.read_index(CONFIG.faiss_index_path)
logger.info("Loading chunks...")
with open(CONFIG.chunks_path, "rb") as f:
    chunks = pickle.load(f)
logger.info("All resources loaded successfully")

# Pydantic model for input validation
class QueryRequest(BaseModel):
    q: str

# Response model
class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    latency: Dict[str, float]
    follow_up_questions: List[str]

def retrieve_chunks(query: str, k: int = CONFIG.retrieve_k) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks from FAISS index."""
    start_time = time.time()
    query_embedding = embedder.encode([query])
    latency_embedding = time.time() - start_time

    start_time = time.time()
    distances, chunk_ids = index.search(query_embedding, k)
    latency_search = time.time() - start_time

    valid_indices = [i for i in chunk_ids[0] if 0 <= i < len(chunks)]
    if not valid_indices:
        logger.warning("No valid chunks found")
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    candidates = [chunks[i] for i in valid_indices]
    return candidates, {
        "embedding": latency_embedding,
        "vector_search": latency_search
    }

def rerank_chunks(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank chunks using cross-encoder."""
    start_time = time.time()
    pairs = [[query, chunk.page_content] for chunk in candidates]
    scores = reranker.predict(pairs)
    ranked_results = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)[:CONFIG.top_k]
    latency_reranking = time.time() - start_time
    return [chunk for _, chunk in ranked_results], {"reranking": latency_reranking}

def get_llm_response(context: str, question: str, model: str) -> str:
    """Get response from Ollama API."""
    context = context[:CONFIG.max_context_chars]
    logger.info(f"Context length (characters): {len(context)}")

    payload = {
        "model": model or CONFIG.llm_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the provided context to answer the question in a clear, concise paragraph. If the context lacks details, provide a brief answer based on available information."
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        "stream": False,
        "options": {"num_ctx": 4096, "temperature": CONFIG.temperature}
    }

    try:
        response = requests.post(
            CONFIG.ollama_url, json=payload, timeout=CONFIG.llm_timeout
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return content or "Sorry, I couldn't find enough information to answer."
    except Exception as e:
        logger.error(f"Ollama error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

def generate_follow_up_questions(query: str, answer: str, context: str) -> List[str]:
    """Generate follow-up questions using LLM."""
    prompt = f"""
    Based on the following question, answer, and context, generate {CONFIG.num_follow_up_questions} natural follow-up questions:

    Question: {query}
    Answer: {answer}
    Context: {context[:500]}...

    Follow-up questions:
    """

    payload = {
        "model": CONFIG.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": CONFIG.temperature}
    }

    try:
        response = requests.post(
            CONFIG.ollama_url, json=payload, timeout=CONFIG.llm_timeout
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        questions = [
            line.split(".", 1)[-1].split(")", 1)[-1].strip()
            for line in content.split("\n")
            if line and (line[0].isdigit() or line.startswith("-")) and "?" in line
        ]
        return questions[:CONFIG.num_follow_up_questions]
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {str(e)}")
        return []

def track_metrics(
    query: str,
    retrieved_indices: List[int],
    relevant_indices: List[int],
    latency_data: Dict[str, float],
    rag_answer: str,
    direct_answer: str
):
    """Track metrics with MLflow."""
    with mlflow.start_run():
        # Precision@k and Recall@k
        k = len(retrieved_indices)
        if relevant_indices and retrieved_indices:
            relevant_retrieved = set(retrieved_indices).intersection(set(relevant_indices))
            precision_at_k = len(relevant_retrieved) / k if k > 0 else 0
            recall_at_k = len(relevant_retrieved) / len(relevant_indices) if relevant_indices else 0
            mlflow.log_metric("precision_at_k", precision_at_k)
            mlflow.log_metric("recall_at_k", recall_at_k)

        # Latency metrics
        for key, value in latency_data.items():
            mlflow.log_metric(f"latency_{key}", value)

        # BLEU and ROUGE scores
        if rag_answer and direct_answer:
            smooth = SmoothingFunction().method1
            reference = [direct_answer.split()]
            candidate = rag_answer.split()
            bleu_score = sentence_bleu(reference, candidate, smoothing_function=smooth)
            mlflow.log_metric("bleu_score", bleu_score)

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(direct_answer, rag_answer)
            mlflow.log_metric("rouge_1_f", scores['rouge1'].fmeasure)
            mlflow.log_metric("rouge_2_f", scores['rouge2'].fmeasure)
            mlflow.log_metric("rouge_l_f", scores['rougeL'].fmeasure)

        mlflow.log_param("query", query)


@app.post("/ask", response_model=QueryResponse)
async def ask(q: str = Form(...)):  # Changed from request: QueryRequest to q: str = Form(...)
    """Handle query and return RAG response."""
    logger.info(f"Received query: {q}")
    latency_data = {}

    # Retrieve and rerank chunks
    try:
        candidates, retrieval_latency = retrieve_chunks(q)  # Changed from request.q
        ranked_chunks, reranking_latency = rerank_chunks(q, candidates)  # Changed from request.q
        latency_data.update(retrieval_latency)
        latency_data.update(reranking_latency)
    except HTTPException as e:
        return QueryResponse(
            answer=e.detail,
            sources=[],
            latency=latency_data,
            follow_up_questions=[]
        )

    # Generate answer
    context = "\n\n".join(chunk.page_content for chunk in ranked_chunks)
    start_time = time.time()
    answer = get_llm_response(context, q, CONFIG.llm_model)  # Changed from request.q
    latency_data["llm_response"] = time.time() - start_time

    # Get direct answer for comparison
    comparison_answer = get_llm_response(context, q, CONFIG.comparison_model)  # Changed from request.q

    # Generate follow-up questions
    follow_up_questions = generate_follow_up_questions(q, answer, context)  # Changed from request.q

    # Track metrics
    valid_indices = [i for i in range(len(chunks)) if chunks[i] in candidates]
    # Replace the simulated relevance line with this more meaningful approach
    # Determine relevance based on semantic similarity threshold
    query_embedding = embedder.encode([q])
    similarities = [np.dot(query_embedding[0], embedder.encode([chunks[i].page_content])[0]) for i in valid_indices]
    similarity_threshold = 0.65  # Adjust threshold as needed
    relevant_indices = [valid_indices[i] for i, sim in enumerate(similarities) if sim > similarity_threshold]
    if not relevant_indices:  # Ensure we have at least one relevant document
        relevant_indices = [valid_indices[np.argmax(similarities)]]
    track_metrics(q, valid_indices, relevant_indices, latency_data, answer, comparison_answer)  # Changed from request.q

    return QueryResponse(
        answer=answer,
        sources=[chunk.metadata for chunk in ranked_chunks],
        latency=latency_data,
        follow_up_questions=follow_up_questions
    )
# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    """Serve the index.html page."""
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        reload_dirs=["./"],  # Only watch specific directories
        log_level="debug"    # More detailed logging during development
    )