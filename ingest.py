import torch  # Tool to check if GPU is available
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Tool to break text into smaller pieces
from sentence_transformers import SentenceTransformer  # Tool to turn text into numbers (vectors) for searching
import faiss  # Tool for fast searching of those numbers (vectors)
import pickle  # Tool to save Python data to files, like saving a game
import sys  # Tool to get information from the command line (e.g., the PDF file name)
import fitz  # Tool (PyMuPDF) to open and read PDF files
from langchain.schema import Document  # A special container to hold text and extra info (like page numbers)

# Get the PDF file name from the command line
pdf_path = sys.argv[1]
print(f"Starting to process the PDF file: {pdf_path}...")

# Check if GPU is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# embedder: A tool that turns text into numbers (embeddings) for searching
# Using a pre-trained model "all-MiniLM-L6-v2" for good text understanding
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

# splitter: A tool that cuts big text into smaller chunks based on characters
# Note: chunk_size=400 means 400 characters (not tokens), adjust if needed
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

# doc: The whole PDF file loaded into memory
doc = fitz.open(pdf_path)

# all_chunks: A list to keep all the small text pieces from the PDF
all_chunks = []

# index: A special storage for the embeddings
index = None

# Process each page of the PDF one by one
for page_num in range(len(doc)):
    # Load the current page
    page = doc.load_page(page_num)

    # Get the text from the page
    page_text = page.get_text()

    # Put the page text into a Document object with metadata
    doc_obj = Document(page_content=page_text, metadata={"page": page_num, "source": pdf_path})

    # Cut the page text into smaller chunks
    page_chunks = splitter.split_documents([doc_obj])

    # Check if we got any chunks from this page
    if page_chunks:
        # Extract the text from each chunk
        texts = [chunk.page_content for chunk in page_chunks]

        # Turn the text into embeddings with a larger batch size for speed
        vecs = embedder.encode(texts, batch_size=128, show_progress_bar=True)

        # Set up the FAISS index if not initialized
        if index is None:
            embedding_dimension = vecs.shape[1]
            index = faiss.IndexFlatIP(embedding_dimension)

        # Add the embeddings to the index
        index.add(vecs)

        # Add the chunks to the list
        all_chunks.extend(page_chunks)

    # Show progress
    print(f"Finished page {page_num+1} out of {len(doc)}, made {len(page_chunks)} chunks")

# Close the PDF file
doc.close()

# Save everything we made
if index is not None:
    # Verify the number of chunks and vectors match
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Total vectors in index: {index.ntotal}")

    # Save the index and chunks
    faiss.write_index(index, "index.faiss")
    pickle.dump(all_chunks, open("chunks.pkl", "wb"))
    print(f"✅ Done! Saved {len(all_chunks)} chunks")
else:
    print("⚠️ Oops! No chunks were made from the PDF!")