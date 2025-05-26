from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import os

from src.models.schemas import ChatRequest
from src.core.rag_system import RAGSystem

app = FastAPI(title="RAG API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system at module level

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
