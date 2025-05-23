from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from langchain_core.documents import Document

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use.
                      Default is a fast and efficient model trained on MS MARCO.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def rerank_documents(
        self,
        query: str,
        documents: List[Document], 
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """Rerank documents using cross-encoder scores.
        
        Args:
            query: The search query
            documents: List of retrieved documents to rerank
            top_k: Number of top documents to return. If None, returns all reranked.
        
        Returns:
            List of (document, score) tuples sorted by relevance score.
        """
        # Prepare text pairs for cross-encoder
        text_pairs = [(query, doc.page_content) for doc in documents]
        
        # Get cross-encoder scores in batches
        batch_size = 8  # Adjust based on available memory
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(text_pairs), batch_size):
                batch = text_pairs[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                scores.extend(batch_scores)
                
        # Create document-score pairs and sort by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            doc_score_pairs = doc_score_pairs[:top_k]
            
        return doc_score_pairs
