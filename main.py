from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
from src.document_processing.processor import DocumentProcessor
from src.core.vectorstore import VectorStore
from src.core.llm import init_llm
from src.core.reranking import CrossEncoderReranker
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import shutil
import os
import re

app = FastAPI(title="RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    conversation_id: str

class RAGSystem:    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm = init_llm()
        self.qa_chain = None
        self._current_search_type = None
        self.reranker = CrossEncoderReranker()
        self.conversations = {}  # Store conversations by ID
        
    def get_prompt_templates(self) -> tuple[PromptTemplate, PromptTemplate]:
        """Generate the prompt templates based on the current document."""
        doc_name = self.vector_store.get_source_document_name() or "the provided document"
        
        initial_template = f"""You are an expert helping developers understand concepts from {doc_name}.
Use the following context to answer the question. Please:
- Break down complex concepts into digestible parts
- Use bullet points for clarity where appropriate
- Include relevant code examples when helpful
- Keep explanations precise but thorough
- Cite page numbers and relevant sections when available
- If you don't know the answer, say so instead of making it up

Context: {{context}}
Question: {{question}}

Answer: Let's explain this step by step:"""

        refine_template = f"""You are an expert helping developers understand concepts from {doc_name}.

Here's the original question: {{question}}

We have already provided the following explanation:
{{existing_answer}}

Now we have found some additional context: {{context}}

Please refine the explanation by:
1. Adding any new important information
2. Correcting any inaccuracies
3. Maintaining clear formatting with bullet points
4. Including page numbers and citations where available
5. Adding relevant code examples if new ones are found

Updated answer:"""

        initial_prompt = PromptTemplate(
            template=initial_template,
            input_variables=["context", "question"]
        )
        
        refine_prompt = PromptTemplate(
            template=refine_template,
            input_variables=["question", "existing_answer", "context"]
        )
        
        return initial_prompt, refine_prompt

    def process_and_index_document(self, file_path: Path) -> int:
        """Process a document and index it in the vector store."""
        chunks = self.document_processor.process_document(file_path)
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        self.vector_store.create_vector_store(texts, metadatas)
        self.vector_store.save_vector_store()
        return len(chunks)      
    def setup_qa_chain(self, conversation_id: Optional[str] = None):
        """Initialize the QA chain with the vector store retriever."""
        if not self.vector_store.vector_store:
            self.vector_store.load_vector_store()
        
        if not self.vector_store.vector_store:
            raise ValueError("No vector store available. Please index documents first.")

        # Use memory for chat sessions
        memory = self.conversations.get(conversation_id) if conversation_id else None

        # Retrieve more documents initially for better reranking
        base_retriever = self.vector_store.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Get more documents for reranking
        )

        if conversation_id:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                chain_type="stuff",
                retriever=base_retriever,
                memory=memory,
                return_source_documents=True,
                verbose=False
            )
        else:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=base_retriever,
                return_source_documents=True
            )    
    def expand_query(self, question: str) -> list[str]:
        """Generate related questions for multiple query expansion."""
        expansion_prompt = f"""Given the user question: "{question}"

Please generate 5 related but more specific questions that would help provide a comprehensive answer.
Return only the questions as a numbered list without any introduction or explanation."""

        expansion_response = self.llm.invoke(expansion_prompt)
        content = expansion_response.content if hasattr(expansion_response, 'content') else str(expansion_response)
        
        expanded_questions = []
        for q in content.split('\n'):
            q = q.strip()
            if q and any(c.isdigit() for c in q[:2]):
                q = re.sub(r'^\d+\.\s*', '', q)
                expanded_questions.append(q)
        # Print expanded queries for visibility
        print("\nGenerated expanded queries:")
        for i, q in enumerate(expanded_questions, 1):
            print(f"{i}. {q}")
            print()
        
        return [question] + expanded_questions      
    def query_document(self, query: str, conversation_id: str) -> tuple[str, list]:
        """Query the document using multiple query expansion and cross-encoder reranking."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                input_key="question"
            )
        self.setup_qa_chain(conversation_id=conversation_id)

        # Generate expanded queries (fixed at 5 queries)
        expanded_queries = self.expand_query(query)
        all_docs = []
        seen_content = set()
        
        # Process each expanded query to gather relevant documents
        for expanded_q in expanded_queries:
            response = self.qa_chain.invoke({"question": expanded_q})
            if 'source_documents' in response:
                for doc in response['source_documents']:
                    if doc.page_content not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(doc.page_content)
        
        # Apply cross-encoder reranking with relevance scores
        reranked_docs = self.reranker.rerank_documents(
            query=query,
            documents=all_docs,
            top_k=10  # Keep more relevant documents
        )
        
        # Prepare comprehensive context from reranked documents with scores
        reranked_context = "\n\n".join([
            f"[Passage {idx+1}, Relevance: {score:.3f}]\n{doc.page_content}\n[Source: Page {doc.metadata.get('page', 'Unknown')}, Chunk {doc.metadata.get('chunk', 'Unknown')}]"
            for idx, (doc, score) in enumerate(reranked_docs)
        ])        # Create a detailed prompt that encourages comprehensive answers
        detailed_prompt = f"""Based on the following passages ranked by relevance to the question: "{query}"

{reranked_context}

Please provide a comprehensive answer that:
1. Synthesizes information from all relevant passages
2. Uses clear examples and relevant quotes
3. Breaks down complex concepts into easy-to-understand parts
4. Uses a clear structure with sections and bullet points
5. Provides code examples if relevant
6. Maintains natural flow between concepts

Important: Do not mention page numbers, chunks, or source references in your answer.
Focus on delivering the information in a clear, user-friendly way.
If certain passages contradict each other, acknowledge this and explain the different perspectives.
Previous conversation context should be considered for a coherent dialogue.

Question: {query}
"""
        # Generate final response using conversation memory
        conversation_history = self.conversations[conversation_id].chat_memory.messages
        final_response = self.llm.invoke(detailed_prompt + "\n\nPrevious conversation context:\n" + 
                                       str(conversation_history) if conversation_history else "")
        
        answer = final_response.content if hasattr(final_response, 'content') else str(final_response)
        
        # Update conversation memory
        self.conversations[conversation_id].save_context(
            {"question": query},
            {"answer": answer}
        )
            
        return answer, [doc for doc, _ in reranked_docs]

    def clear_vector_store(self):
        """Clear the vector store and its saved files."""
        if self.vector_store:
            self.vector_store.vector_store = None
            index_dir = Path("data/processed/faiss_index")
            if index_dir.exists():
                shutil.rmtree(index_dir)

# Initialize RAG system
rag_system = RAGSystem()

@app.post("/upload")
async def upload_document(file: UploadFile):
    """Upload and index a document."""
    try:
        # Save the uploaded file
        file_path = Path(f"documents/{file.filename}")
        os.makedirs(file_path.parent, exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process and index the document
        num_chunks = rag_system.process_and_index_document(file_path)
        
        return {"message": f"Successfully processed and indexed {num_chunks} chunks from {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """Query the document with conversation history."""
    try:
        answer, source_docs = rag_system.query_document(
            query=request.query,
            conversation_id=request.conversation_id
        )
        
        sources = [
            {
                "page": doc.metadata.get("page", "Unknown"),
                "chunk": doc.metadata.get("chunk", "Unknown"),
                "source": doc.metadata.get("source", "Unknown")
            }
            for doc in source_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_index():
    """Clear the vector store and indexed documents."""
    try:
        rag_system.clear_vector_store()
        return {"message": "Vector store cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
