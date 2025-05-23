from typing import List, Optional
import os
from pathlib import Path
from langchain_community.vectorstores.faiss import FAISS
from src.config.settings import VECTOR_STORE_PATH
from src.core.llm import init_embeddings

class VectorStore:
    def __init__(self):
        self.embeddings = init_embeddings()
        self.vector_store = None

    def create_vector_store(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Create a new vector store from the given texts."""
        self.vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        return self.vector_store

    def save_vector_store(self):
        """Save the vector store to disk."""
        if self.vector_store:
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
            self.vector_store.save_local(VECTOR_STORE_PATH)

    def load_vector_store(self) -> Optional[FAISS]:
        """Load the vector store from disk."""
        if os.path.exists(VECTOR_STORE_PATH):
            self.vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True  # Only use this if you trust the source of the vector store
            )
            return self.vector_store
        return None

    def similarity_search(self, query: str, k: int = 4) -> List[dict]:
        """Perform similarity search on the vector store."""
        if not self.vector_store:
            self.load_vector_store()
            if not self.vector_store:
                raise ValueError("No vector store available. Please create one first.")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in results
        ]

    def get_source_document_name(self) -> Optional[str]:
        """Get the name of the source document from the vector store metadata."""
        if not self.vector_store:
            self.load_vector_store()
        if not self.vector_store:
            return None
            
        # Get first document's metadata to extract source
        results = self.vector_store.similarity_search_with_score("", k=1)
        if results and len(results) > 0:
            doc, _ = results[0]
            if doc.metadata and "source" in doc.metadata:
                return Path(doc.metadata["source"]).name
        return None
