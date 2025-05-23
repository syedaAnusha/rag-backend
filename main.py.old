import click
from pathlib import Path
from src.document_processing.processor import DocumentProcessor
from src.core.vectorstore import VectorStore
from src.core.llm import init_llm
from src.core.reranking import CrossEncoderReranker
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from rich.console import Console
from rich.panel import Panel
import re

console = Console()

class RAGSystem:    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm = init_llm()
        self.qa_chain = None
        self._current_search_type = None
        self.reranker = CrossEncoderReranker()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
        
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

    def process_and_index_document(self, file_path: Path):
        """Process a document and index it in the vector store."""
        # Process the document
        chunks = self.document_processor.process_document(file_path)
        
        # Create vectors and store them
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        self.vector_store.create_vector_store(texts, metadatas)
        self.vector_store.save_vector_store()
        return len(chunks)      
    def setup_qa_chain(self, k: int = 4, search_type: str = "similarity", is_chat: bool = False):
        """Initialize the QA chain with the vector store retriever.
        
        Args:
            k (int): Number of documents to retrieve
            search_type (str): Type of search to use. Either "similarity" or "mmr"
            is_chat (bool): Whether this is a chat session (True) or one-off query (False)
        """
        if not self.vector_store.vector_store:
            self.vector_store.load_vector_store()
        
        if not self.vector_store.vector_store:
            raise ValueError("No vector store available. Please index documents first.")        # Set up the base retriever according to search type
        if search_type == "mmr":
            base_retriever = self.vector_store.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": k*2,  # Fetch more documents for MMR to choose from
                    "lambda_mult": 0.5  # Control diversity
                }
            )
        elif search_type == "multiple-query":
            # For multiple-query, we'll use similarity search but handle expansion in query_document
            base_retriever = self.vector_store.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        else:  # similarity search or chat mode
            base_retriever = self.vector_store.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        
        # Add contextual compression only for one-off MMR queries
        if not is_chat and search_type == "mmr":
            # Create a compressor that uses LLM to extract relevant information
            compressor = LLMChainExtractor.from_llm(self.llm)
            
            # Wrap the base retriever with contextual compression
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            # For one-off MMR queries, use RetrievalQA with refine chain type
            initial_prompt, refine_prompt = self.get_prompt_templates()
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="refine",
                retriever=retriever,
                chain_type_kwargs={
                    "question_prompt": initial_prompt,
                    "refine_prompt": refine_prompt,
                    "document_variable_name": "context"
                },
                return_source_documents=True
            )
        else:
            retriever = base_retriever
            # For similarity search, use ConversationalRetrievalChain with stuff chain type
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
        return self.qa_chain            
    def expand_query(self, question: str, num_queries: int = 3) -> list[str]:
        """Generate related questions for multiple query expansion.
        
        Args:
            question (str): The original question to expand
            num_queries (int): Number of additional questions to generate
            
        Returns:
            list[str]: List of questions including the original and generated ones
        """
        expansion_prompt = f"""Given the user question: "{question}"

Please generate {num_queries} related but more specific questions that would help provide a comprehensive answer.
Return only the questions as a numbered list without any introduction or explanation."""

        expansion_response = self.llm.invoke(expansion_prompt)
        content = expansion_response.content if hasattr(expansion_response, 'content') else str(expansion_response)
          # Extract numbered questions from the response and clean them
        expanded_questions = []
        for q in content.split('\n'):
            q = q.strip()
            if q and any(c.isdigit() for c in q[:2]):
                # Remove the numbering prefix (e.g., "1. ", "2. ", etc.)
                q = re.sub(r'^\d+\.\s*', '', q)
                expanded_questions.append(q)
        
        # Print expanded queries for visibility
        console.print("\n[blue]Generated expanded queries:[/blue]")
        for i, q in enumerate(expanded_questions, 1):
            console.print(f"[cyan]{i}.[/cyan] {q}")
        console.print()
        
        # Include original question
        return [question] + expanded_questions      
    def query_document(self, query: str, k: int = 4, search_type: str = "similarity", is_chat: bool = False,
                    num_queries: int = 3, inner_search_type: str = "similarity"):
        """Query the document using QA chain for contextual answers."""
        if not self.qa_chain or self._current_search_type != search_type:
            self.setup_qa_chain(k=k, search_type=search_type, is_chat=is_chat)
            self._current_search_type = search_type

        if search_type == "multiple-query":
            # Generate expanded queries
            expanded_queries = self.expand_query(query, num_queries)
            all_docs = []
            seen_content = set()
            
            # Configure retriever based on chosen inner search type
            if inner_search_type == "mmr":
                base_retriever = self.vector_store.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": max(2, k // (num_queries + 1)),
                        "fetch_k": k,
                        "lambda_mult": 0.5
                    }
                )
            else:  # similarity search
                base_retriever = self.vector_store.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": max(2, k // (num_queries + 1))}
                )
            
            # Use single chain for all queries
            temp_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=base_retriever,
                return_source_documents=True
            )
            
            # Process each query
            console.print(f"\n[blue]Running {inner_search_type} search for queries...[/blue]")
            all_docs = []
            seen_content = set()
            
            # First pass: collect documents from each expanded query
            for expanded_q in expanded_queries:
                response = temp_chain.invoke({"query": expanded_q})
                if 'source_documents' in response:
                    for doc in response['source_documents']:
                        if doc.page_content not in seen_content:
                            all_docs.append(doc)
                            seen_content.add(doc.page_content)
            
            # Apply cross-encoder reranking
            console.print("[blue]Reranking documents with cross-encoder...[/blue]")
            reranked_docs = self.reranker.rerank_documents(
                query=query,  # Use original query for final ranking
                documents=all_docs,
                top_k=k  # Only keep top k documents after reranking
            )
            
            # Update all_docs with reranked documents
            all_docs = [doc for doc, _ in reranked_docs]  # Discard scores after reranking
            
            # Generate visualization if we have results
            if all_docs:
                try:
                    from src.utils.visualization import plot_query_document_space
                    import numpy as np
                    from pathlib import Path                    # Get all document chunks and their embeddings from the vector store
                    corpus_docs = list(self.vector_store.vector_store.docstore._dict.values())
                    
                    # Get the actual embeddings for corpus documents from FAISS
                    console.print("[blue]Getting corpus embeddings from FAISS...[/blue]")
                    corpus_embeddings = self.vector_store.vector_store.index.reconstruct_n(0, len(corpus_docs))
                    
                    # Prepare query and document texts for embedding
                    texts_to_embed = (
                        [query] +  # Original query only
                        [doc.page_content for doc in all_docs]  # Then retrieved docs
                    )
                    
                    # Batch embed query and retrieved documents
                    console.print("[blue]Computing embeddings for queries and retrieved documents...[/blue]")
                    new_embeddings = self.vector_store.embeddings.embed_documents(texts_to_embed)
                    
                    # Split embeddings into groups - only original query
                    query_embeddings = np.array([new_embeddings[0]])  # Just the original query
                    doc_embeddings = np.array(new_embeddings[1:])
                    
                    # Create visualizations directory
                    plots_dir = Path("data/visualizations")
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate and save the plot
                    plot_query_document_space(
                        queries=expanded_queries,
                        query_embeddings=query_embeddings,
                        documents=all_docs,
                        doc_embeddings=doc_embeddings,
                        corpus_docs=corpus_docs,
                        corpus_embeddings=corpus_embeddings,
                        output_dir=plots_dir
                    )
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not generate visualization: {str(e)}[/yellow]")
            
                # Create final chain for comprehensive answer
                final_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="refine",
                    retriever=base_retriever,
                    chain_type_kwargs={
                        "question_prompt": self.get_prompt_templates()[0],
                        "refine_prompt": self.get_prompt_templates()[1],
                        "document_variable_name": "context"
                    },
                    return_source_documents=True
                )
                
                # Generate final response
                response = final_chain.invoke({
                    "query": query,
                    "chat_history": [{"question": q} for q in expanded_queries[1:]]
                })
                answer = response["result"] if "result" in response else response["answer"]
                response["source_documents"] = all_docs
            else:
                # Fallback to basic retrieval if no expanded results
                response = self.qa_chain.invoke({"question": query})
                answer = response["answer"]
        else:
            # For non-multiple-query searches
            if not is_chat and search_type == "mmr":
                response = self.qa_chain.invoke({"query": query})
                answer = response["result"]
            else:
                response = self.qa_chain.invoke({"question": query})
                answer = response["answer"]
            
        return answer, response["source_documents"]

@click.group()
def cli():
    """RAG CLI System for document querying."""
    pass

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def index(file_path):
    """Index a document for searching."""
    rag = RAGSystem()
    file_path = Path(file_path)
    num_chunks = rag.process_and_index_document(file_path)
    console.print(f"\n[green]Successfully processed and indexed {num_chunks} chunks from {file_path}[/green]")

@cli.command()
@click.argument('query')
@click.option('--k', default=4, help='Number of document chunks to use for generating the answer')
@click.option('--search-type', 
              type=click.Choice(['similarity', 'mmr', 'multiple-query']), 
              default='similarity',
              help='Search technique to use. "similarity" for basic search, "mmr" for diversity, "multiple-query" for comprehensive results')
@click.option('--num-queries', default=3, 
              help='Number of additional queries to generate when using multiple-query search type')
@click.option('--inner-search-type',
              type=click.Choice(['similarity', 'mmr']),
              default='similarity',
              help='Search technique to use for expanded queries when using multiple-query search type. In multiple-query mode, results are always reranked using cross-encoder.')
def search(query, k, search_type, num_queries, inner_search_type):
    """Search and answer questions using the indexed documents."""
    rag = RAGSystem()
    answer, source_docs = rag.query_document(
        query, 
        k=k, 
        search_type=search_type, 
        is_chat=False,
        num_queries=num_queries,
        inner_search_type=inner_search_type
    )
    
    # Print the query
    console.print("\n" + "="*80)
    console.print("[blue]🔍 Question: [/blue]" + f"[green bold]\"{query}\"[/green bold]")
    console.print("="*80)
    
    # Print the answer in a panel with markdown formatting
    answer_panel = Panel(
        answer,
        title="[yellow bold]Answer[/yellow bold]",
        border_style="yellow",
        padding=(1, 2)
    )
    console.print("\n", answer_panel)
    
    # Print source information
    if source_docs:
        # Get the source document name (should be same for all chunks)
        source = source_docs[0].metadata.get('source', 'Unknown source')
        source_name = Path(source).name
        
        # Print references with source name only once
        console.print("\n[cyan bold]References from [/cyan bold]" + f"[cyan]📚 {source_name}:[/cyan]")
        for i, doc in enumerate(source_docs, 1):
            page = doc.metadata.get('page', 'Unknown')
            chunk = doc.metadata.get('chunk', 'Unknown')
            console.print(f"[cyan]{i}.[/cyan] Page {page}, Chunk {chunk}")
    
    console.print("\n" + "="*80)

@cli.command()
@click.option('--k', default=4, help='Number of document chunks to use for generating the answer')
@click.option('--search-type', 
              type=click.Choice(['similarity', 'mmr', 'multiple-query']), 
              default='similarity',
              help='Search technique to use. "similarity" for basic search, "mmr" for diversity, "multiple-query" for comprehensive results')
@click.option('--num-queries', default=3, 
              help='Number of additional queries to generate when using multiple-query search type')
@click.option('--inner-search-type',
              type=click.Choice(['similarity', 'mmr']),
              default='similarity',
              help='Search technique to use for expanded queries when using multiple-query search type')
def chat(k, search_type, num_queries, inner_search_type):
    """Start an interactive chat session with memory of conversation history."""
    rag = RAGSystem()
    console.print("\n[green]Starting chat session. Type 'exit' to end the conversation.[/green]")
    console.print("[green]The system will remember your conversation history.[/green]")
    console.print(f"[green]Using {search_type} search with {k} documents.[/green]\n")
    
    while True:
        # Get user input
        query = input(click.style("You: ", fg='green', bold=True))
        
        # Check for exit command
        if query.lower() in ['exit', 'quit']:
            console.print("\n[yellow]Ending chat session...[/yellow]")
            break
          # Get response
        answer, source_docs = rag.query_document(
            query, 
            k=k, 
            search_type=search_type, 
            is_chat=True,
            num_queries=num_queries,
            inner_search_type=inner_search_type
        )
        
        # Print the answer
        answer_panel = Panel(
            answer,
            title="[yellow bold]Assistant[/yellow bold]",
            border_style="yellow",
            padding=(1, 2)
        )
        console.print("\n", answer_panel)
        
        # Print source information
        if source_docs:
            source = source_docs[0].metadata.get('source', 'Unknown source')
            source_name = Path(source).name
            
            # Print references with source name only once
            console.print("\n[cyan bold]References from [/cyan bold]" + f"[cyan]📚 {source_name}:[/cyan]")
            for i, doc in enumerate(source_docs, 1):
                page = doc.metadata.get('page', 'Unknown')
                chunk = doc.metadata.get('chunk', 'Unknown')
                console.print(f"[cyan]{i}.[/cyan] Page {page}, Chunk {chunk}")
        
        console.print("\n" + "-"*80 + "\n")

if __name__ == '__main__':
    cli()