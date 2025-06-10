# RagJet: Streamlined Retrieval-Augmented Generation Framework

A production-ready RAG system with semantic search, document reranking, and comprehensive performance tracking.

## Overview

RagJet is a framework for building and evaluating Retrieval-Augmented Generation (RAG) applications. It efficiently chunks documents, creates vector embeddings, performs semantic search, reranks results, and generates high-quality responses using local LLMs via Ollama.

## Key Features

- **Semantic Document Retrieval**: Uses FAISS for efficient vector search
- **Cross-Encoder Reranking**: Refines retrieved chunks for higher precision
- **Automated Index Management**: Detects when reindexing would improve performance
- **Comprehensive Metrics**: Tracks retrieval precision, generation quality, and latency
- **Web Interface**: Simple UI for querying documents and viewing responses
- **Auto-Reload Development**: Automatic server refresh during development

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ragjet.git
cd ragjet

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install and start Ollama (for LLM inference)
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
ollama pull llama3.2:1b-instruct-q3_K_S
```

## Quick Start

1. **Index your documents**:
```bash
python ingest.py your_document.pdf
```

2. **Start the server**:
```bash
python app.py
```

3. **Access the web interface** at http://localhost:8000

4. **Manage your index**:
```bash
python embedding_manager.py
```

## Usage

### Document Processing

Process PDF documents into searchable chunks:

```bash
python ingest.py Pride_and_Prejudice.pdf
```

### Index Management

The embedding manager provides utilities for maintaining your vector index:

```bash
python embedding_manager.py
```

Options:
1. Add mock documents (for testing)
2. Check if reindex is needed
3. Perform manual reindex
4. Do all of the above

### Web Interface

Start the FastAPI server and access the web UI:

```bash
python app.py
```

Features:
- Ask questions about your documents
- View source citations
- See performance metrics
- Get follow-up question suggestions

## Architecture

RagJet implements a multi-step RAG pipeline:

1. **Document Processing**: Chunks documents using RecursiveCharacterTextSplitter
2. **Embedding Generation**: Creates vector representations using Sentence Transformers
3. **Vector Search**: Efficiently retrieves semantically relevant chunks with FAISS
4. **Reranking**: Uses cross-encoder model to refine results
5. **Context Assembly**: Combines relevant chunks into coherent context
6. **Response Generation**: Leverages Ollama models to generate accurate answers
7. **Follow-up Suggestion**: Automatically generates related questions

## Performance Metrics

RagJet tracks comprehensive metrics to evaluate and improve performance:

| Metric | Description | Good Range |
|--------|-------------|------------|
| Precision@k | Percentage of relevant documents retrieved | 0.2-0.8 |
| Recall@k | Percentage of all relevant documents found | 0.5-1.0 |
| ROUGE-1/2/L | Text overlap between generated and reference answers | 0.2-0.6 |
| BLEU Score | N-gram precision score | 0.01-0.3 |
| Latency | Time measurements for each pipeline step | Variable |

View detailed metrics in MLflow:

```bash
mlflow ui
```

Access at http://localhost:5000

## Configuration

Key configuration options in `app.py`:

```python
@dataclass
class Config:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    llm_model: str = "llama3.2:1b-instruct-q3_K_S"
    comparison_model: str = "llama3.2:1b"
    chunks_path: str = "chunks.pkl"
    faiss_index_path: str = "index.faiss"
    max_context_chars: int = 12000
    temperature: float = 0.7
    top_k: int = 5
    retrieve_k: int = 10
    llm_timeout: int = 240
    num_follow_up_questions: int = 3
    ollama_url: str = "http://localhost:11434/api/chat"
```

## Index Management

RagJet provides intelligent index management based on semantic analysis:

- **Semantic Diversity**: Measures how diverse your document collection is
- **Cluster Balance**: Analyzes if documents form well-balanced semantic groups
- **Growth Factor**: Considers index size for performance optimization

The system recommends reindexing when content becomes semantically complex or imbalanced.

## File Structure

```
ragjet/
├── app.py                 # Main FastAPI application
├── ingest.py             # Document processing and indexing
├── embedding_manager.py   # Index management utilities
├── static/
│   └── index.html        # Web interface
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore patterns
```

## Development

To contribute to RagJet:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `python -m pytest`
5. Submit a pull request

The application supports auto-reload during development - changes to code will automatically restart the server.

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**: Ensure Ollama is running on port 11434
2. **Empty Index Error**: Run the ingestion script first to populate your index
3. **Model Download Issues**: Check your internet connection and model cache directory

### Performance Tips

- Use smaller models for faster inference (e.g., `llama3.2:1b` instead of larger variants)
- Adjust `chunk_size` and `chunk_overlap` for your document types
- Monitor MLflow metrics to optimize retrieval parameters

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on [LangChain](https://github.com/hwchase17/langchain) for document processing
- Uses [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- Powered by [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for embeddings
- Web interface built with [FastAPI](https://github.com/tiangolo/fastapi)
- Metrics tracking with [MLflow](https://github.com/mlflow/mlflow)
