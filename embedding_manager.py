import os
import sys
import logging
import numpy as np
import time
import faiss
import pickle
import mlflow
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embedding_manager")

# Configuration class
@dataclass
class Config:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_path: str = "index.faiss"
    chunks_path: str = "chunks.pkl"
    max_context_chars: int = 3500
    temperature: float = 0.7
    top_k: int = 5
    retrieve_k: int = 20
    llm_timeout: int = 60
    num_follow_up_questions: int = 3

CONFIG = Config()

# Document class for chunks
class Document:
    """Simple document class that can be pickled."""
    def __init__(self, page_content="", metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Initialize the embedding model
embedder = SentenceTransformer(CONFIG.embedding_model, cache_folder="./model_cache")

# Define mock documents for testing
MOCK_DOCUMENTS = [
    {
        "content": """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.""",
        "metadata": {"title": "Introduction to AI", "page": 1, "source": "mock_ai.pdf"}
    },
    {
        "content": """Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.""",
        "metadata": {"title": "Machine Learning", "page": 2, "source": "mock_ai.pdf"}
    },
    {
        "content": """Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From the perspective of engineering, it seeks to understand and automate tasks that the human visual system can do.""",
        "metadata": {"title": "Computer Vision", "page": 3, "source": "mock_ai.pdf"}
    },
    {
        "content": """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.""",
        "metadata": {"title": "NLP", "page": 4, "source": "mock_ai.pdf"}
    }
]

def ensure_storage_files_exist():
    """Make sure chunks and index files exist and are properly initialized."""
    # Create or repair chunks file
    try:
        if not os.path.exists(CONFIG.chunks_path) or os.path.getsize(CONFIG.chunks_path) == 0:
            logger.info(f"Creating new chunks file at {CONFIG.chunks_path}")
            with open(CONFIG.chunks_path, "wb") as f:
                pickle.dump([], f)
        else:
            # Try to read the file to validate it
            try:
                with open(CONFIG.chunks_path, "rb") as f:
                    pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                # File exists but is corrupted
                logger.warning(f"Chunks file is corrupted. Creating a new one.")
                with open(CONFIG.chunks_path, "wb") as f:
                    pickle.dump([], f)
    except Exception as e:
        logger.error(f"Error ensuring chunks file: {str(e)}")
        # Last resort - create a new file
        with open(CONFIG.chunks_path, "wb") as f:
            pickle.dump([], f)
    
    # Create or repair index file
    try:
        if not os.path.exists(CONFIG.faiss_index_path) or os.path.getsize(CONFIG.faiss_index_path) == 0:
            logger.info(f"Creating new FAISS index at {CONFIG.faiss_index_path}")
            dimension = embedder.get_sentence_embedding_dimension()
            index = faiss.IndexFlatIP(dimension)
            faiss.write_index(index, CONFIG.faiss_index_path)
    except Exception as e:
        logger.error(f"Error ensuring index file: {str(e)}")
        # Last resort - create a new index
        dimension = embedder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dimension)
        faiss.write_index(index, CONFIG.faiss_index_path)

def add_documents(documents: List[Dict], chunks_path: str = CONFIG.chunks_path, 
                 index_path: str = CONFIG.faiss_index_path) -> int:
    """Add new documents to the index and return number of chunks added."""
    # Ensure files exist
    ensure_storage_files_exist()
    
    logger.info(f"Adding {len(documents)} new documents to the index")
    
    # Track start time for metrics
    start_time = time.time()
    
    # Load existing chunks and index
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    index = faiss.read_index(index_path)
    original_count = len(chunks)
    
    # Process each document and create new chunks
    new_chunks = []
    for doc in documents:
        # Create simple document chunk
        chunk = Document(page_content=doc["content"], metadata=doc["metadata"])
        new_chunks.append(chunk)
    
    # Get embeddings for new chunks
    texts = [chunk.page_content for chunk in new_chunks]
    new_embeddings = embedder.encode(texts)
    
    # Add to index and update chunks list
    with mlflow.start_run(run_name="add_documents"):
        # Log basic info
        mlflow.log_param("original_count", original_count)
        mlflow.log_param("new_docs_count", len(documents))
        
        # Add embeddings to index
        index.add(new_embeddings.astype(np.float32))
        
        # Add chunks to list
        chunks.extend(new_chunks)
        
        # Save updated chunks and index
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)
        
        faiss.write_index(index, index_path)
        
        # Log metrics
        processing_time = time.time() - start_time
        mlflow.log_metric("processing_time", processing_time)
        mlflow.log_metric("new_count", len(chunks))
        
        logger.info(f"Added {len(new_chunks)} chunks. Total count: {len(chunks)}")
    
    return len(new_chunks)

def check_reindex_needed(chunks_path: str = CONFIG.chunks_path, 
                        index_path: str = CONFIG.faiss_index_path,
                        threshold: float = 0.25) -> Dict[str, Any]:
    """Check if reindexing is needed based on semantic diversity analysis."""
    # Ensure files exist
    ensure_storage_files_exist()
    
    # Load chunks and index
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    index = faiss.read_index(index_path)
    
    if len(chunks) < 5:
        logger.info("Not enough documents to analyze for reindexing needs")
        return {"needs_reindex": False, "reindex_score": 0.0, "total_chunks": len(chunks)}
    
    # Get a random sample of chunk embeddings (for efficiency with large collections)
    sample_size = min(100, len(chunks))
    sample_indices = np.random.choice(len(chunks), sample_size, replace=False)
    sample_chunks = [chunks[i] for i in sample_indices]
    sample_texts = [chunk.page_content for chunk in sample_chunks]
    
    # Generate embeddings for the sample
    sample_embeddings = embedder.encode(sample_texts)
    
    # 1. Semantic diversity check: Calculate average cosine similarity between samples
    # Higher avg similarity = less diverse = less need to reindex
    similarity_matrix = np.dot(sample_embeddings, sample_embeddings.T)
    np.fill_diagonal(similarity_matrix, 0)  # Exclude self-similarity
    avg_similarity = np.sum(similarity_matrix) / (sample_size * (sample_size - 1))
    
    # Normalize to 0-1 range where higher = more need to reindex
    # Lower similarity (more diverse content) suggests reindexing might help
    semantic_diversity_score = 1.0 - avg_similarity
    
    # 2. Cluster analysis: K-means clustering to detect semantic groups
    n_clusters = min(5, sample_size // 2)
    if n_clusters > 1:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(sample_embeddings)
        
        # Calculate cluster balance (entropy)
        cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)
        cluster_proportions = cluster_counts / sample_size
        cluster_entropy = -np.sum(cluster_proportions * np.log2(cluster_proportions + 1e-10))
        max_entropy = np.log2(n_clusters)
        
        # Lower entropy = more imbalanced = higher reindex need
        cluster_balance_score = 1.0 - (cluster_entropy / max_entropy)
    else:
        cluster_balance_score = 0.0
    
    # 3. Index growth factor
    index_size = index.ntotal
    growth_factor = min(1.0, index_size / 10000)
    
    # Combined score with weights (can be adjusted)
    reindex_score = (
        0.4 * semantic_diversity_score + 
        0.4 * cluster_balance_score + 
        0.2 * growth_factor
    )
    
    needs_reindex = reindex_score > threshold
    
    result = {
        "total_chunks": len(chunks),
        "semantic_diversity": semantic_diversity_score,
        "cluster_balance": cluster_balance_score,
        "growth_factor": growth_factor,
        "reindex_score": reindex_score,
        "needs_reindex": needs_reindex
    }
    
    logger.info(f"Semantic reindex check: score={reindex_score:.3f}, needs_reindex={needs_reindex}")
    return result

def reindex(chunks_path: str = CONFIG.chunks_path, index_path: str = CONFIG.faiss_index_path) -> bool:
    """Rebuild index from scratch."""
    # Ensure files exist
    ensure_storage_files_exist()
    
    logger.info("Starting manual reindexing...")
    start_time = time.time()
    
    # Load chunks
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    
    if not chunks:
        logger.info("No chunks to reindex.")
        return False
    
    # Create new embeddings
    texts = [chunk.page_content for chunk in chunks]
    
    # Process in batches to avoid memory issues with large datasets
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = embedder.encode(batch)
        all_embeddings.append(batch_embeddings)
        logger.info(f"Encoded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
    
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    # Create new index
    dimension = embeddings.shape[1] if len(embeddings) > 0 else embedder.get_sentence_embedding_dimension()
    new_index = faiss.IndexFlatIP(dimension)
    
    if len(embeddings) > 0:
        new_index.add(embeddings.astype(np.float32))
    
    # Save new index
    faiss.write_index(new_index, index_path)
    
    # Log metrics
    duration = time.time() - start_time
    with mlflow.start_run(run_name="manual_reindex"):
        mlflow.log_metric("reindex_time", duration)
        mlflow.log_metric("num_chunks", len(chunks))
    
    logger.info(f"Reindexing complete. Duration: {duration:.2f}s, Chunks: {len(chunks)}")
    return True

def main():
    """Simple command-line interface."""
    print("Document Management Utility")
    print("--------------------------")
    print("1. Add mock documents")
    print("2. Check if reindex is needed")
    print("3. Perform manual reindex")
    print("4. Do all of the above")
    print("5. Exit")
    
    choice = input("Choose an option (1-5): ")
    
    if choice == "1":
        add_documents(MOCK_DOCUMENTS)
        print("Mock documents added successfully!")
    elif choice == "2":
        result = check_reindex_needed()
        print("\nReindex check results:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    elif choice == "3":
        reindex()
        print("Manual reindexing completed!")
    elif choice == "4":
        add_documents(MOCK_DOCUMENTS)
        result = check_reindex_needed()
        print("\nReindex check results:")
        for k, v in result.items():
            print(f"  {k}: {v}")
        if result["needs_reindex"]:
            reindex()
            print("Manual reindexing completed!")
        else:
            print("Reindexing not needed.")
    elif choice == "5":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()