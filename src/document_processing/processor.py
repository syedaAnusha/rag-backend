from typing import List, Dict
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",  # First split by double newlines (paragraphs)
                "\n",    # Then by single newlines
                "(?<=\\.)",  # Split by periods while keeping them
                " ",     # Then by spaces
                ""      # Finally, character by character if needed
            ]
        )

    def get_document_loader(self, file_path: Path):
        """Return appropriate document loader based on file extension."""
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            return PyPDFLoader(str(file_path))
        elif file_ext == '.txt':
            return TextLoader(str(file_path))
        else:
            # Try to use UnstructuredFileLoader for other file types
            try:
                return UnstructuredFileLoader(str(file_path))
            except Exception as e:
                raise ValueError(f"Unsupported file type: {file_ext}. Error: {str(e)}")

    def process_pdf(self, file_path: Path) -> List[Dict]:
        """Process a PDF file and return chunks with metadata."""
        try:
            print(f"\nProcessing PDF: {file_path}")
            
            # Get appropriate loader
            loader = self.get_document_loader(file_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("No content extracted from document")
            
            print(f"Total document pages: {len(documents)}")
            
            # Split documents directly instead of combining
            docs = self.text_splitter.split_documents(documents)
            
            if not docs:
                raise ValueError("Document splitting resulted in no chunks")
            
            #print(f"Created {len(docs)} chunks")
            #print(f"Average chunk size: {sum(len(doc.page_content) for doc in docs) / len(docs):.2f} characters")
            
            return [{
                "text": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "source": str(file_path),
                    "file_type": "pdf",
                    "chunk": idx + 1  # Add chunk number (1-based)
                }
            } for idx, doc in enumerate(docs)]
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise

    def process_document(self, file_path: Path) -> List[Dict]:
        """Process a document based on its file type."""
        if file_path.suffix.lower() == '.pdf':
            return self.process_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
