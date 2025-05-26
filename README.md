# RAG Backend

A sophisticated Retrieval-Augmented Generation (RAG) system built with Python and FastAPI. This system enables intelligent document processing, semantic search, and conversational interactions with documents using state-of-the-art language models.

## Features

- 📚 Document Processing & Indexing
- 🔍 Semantic Search with FAISS
- 💡 Query Expansion for Better Results
- 🔄 Cross-Encoder Reranking
- 💬 Conversational Document Interaction
- 🔗 RESTful API Interface
- 🐳 Docker Support

## Technology Stack

- **Framework**: FastAPI, LangChain
- **Language Models**: LangChain with Google's Generative AI (gemini-2.0-flash)
- **Vector Store**: FAISS
- **Document Processing**: Support for PDF, DOCX, and other text formats
- **Embeddings**: Google Generative AI Embeddings (models/embedding-001)
- **Dependencies**: See `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd rag-backend
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file with:

```env
GOOGLE_API_KEY=your_api_key_here
PORT=8001
```

## API Endpoints

- `POST /upload`: Upload and index a document
- `POST /chat`: Query document with conversation history
- `DELETE /clear`: Clear vector store and indexed documents

## Docker Deployment

1. Build and run using Docker Compose:

```bash
docker-compose up --build
```

The service will be available at `http://localhost:8001`

## Project Structure

```
rag-backend/
├── src/
│   ├── core/              # Core RAG system components
│   ├── document_processing/ # Document processing utilities
│   ├── models/            # Data models and schemas
│   ├── config/            # Configuration settings
│   └── prompts/           # System prompts
├── data/
│   ├── processed/         # Processed document data
│   └── raw/              # Raw document storage
├── documents/            # User uploaded documents
├── main.py              # FastAPI application
├── requirements.txt     # Project dependencies
└── docker-compose.yml   # Docker configuration
```

## Usage Example

1. Start the server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

2. Upload a document:

```bash
curl -X POST "http://localhost:8001/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

3. Query the document:

```bash
curl -X POST "http://localhost:8001/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "conversation_id": "unique_conversation_id"
  }'
```

## Features in Detail

### Document Processing

- Automatic text extraction from PDFs and text files
- Intelligent chunking for optimal context preservation
- Metadata extraction and preservation

### RAG System

- Query expansion for comprehensive answers
- Cross-encoder reranking for improved relevance
- Conversation history management
- Structured response formatting

### Vector Store

- FAISS-based similarity search
- Efficient document indexing
- Persistent storage with backup support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)

## Contact

For any questions or feedback, please open an issue in the repository.
