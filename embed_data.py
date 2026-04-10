import os
import sys
from dotenv import load_dotenv
load_dotenv()

from src.chunking import DocumentStructureChunker
from src.store import EmbeddingStore
from src.models import Document

# Use the real Embedder setup
try:
    from src.embeddings import OpenAIEmbedder
    embedder = OpenAIEmbedder()
except Exception as e:
    from src.embeddings import MockEmbedder
    embedder = MockEmbedder()
    print("Warning: Falling back to MockEmbedder", e)

load_dotenv()

def embed_data():
    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=embedder)
    
    # Note: If CHROMA_PERSIST_DIR is set, and it already has data, 
    # we might want to clear it or make sure we don't duplicate. 
    # Let's just reset by checking size, but Settings(allow_reset=True) doesn't apply to PersistentClient directly here unless we clear the collection.
    
    data_dir = "data"
    chunker = DocumentStructureChunker()
    docs = []
    
    print("Reading and chunking documents...")
    files_processed = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".md"): continue
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
            content = f.read()
            chunks = chunker.chunk(content)
            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}_{i}"
                metadata = {"source": filename, "doc_id": filename}
                docs.append(Document(id=doc_id, content=chunk, metadata=metadata))
            files_processed.append(filename)
                
    if docs:
        print(f"Adding {len(docs)} chunks from {len(files_processed)} files to store...")
        store.add_documents(docs)
        print(f"Store size is now: {store.get_collection_size()} chunks.")
        print(f"ChromaDB persisted successfully to {os.environ.get('CHROMA_PERSIST_DIR', 'in-memory')}")
    else:
        print("No documents found to embed.")

if __name__ == "__main__":
    embed_data()
