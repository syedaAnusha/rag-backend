from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from src.config.settings import GOOGLE_API_KEY, LLM_MODEL, EMBEDDING_MODEL

def init_llm():
    """Initialize the LLM with custom prompt template."""
    prompt_template = """You are a knowledgeable programming expert helping developers understand JavaScript concepts from the book "You Don't Know JS Yet" by Kyle Simpson.

Use the following context to answer the question. Please:
- Respond in a clear, technical tone
- Break down complex concepts into digestible parts
- Use bullet points for clarity where appropriate
- Include relevant code examples when helpful
- Keep explanations precise but thorough
- Cite specific chapters or sections if available
- If the context doesn't contain enough information to answer accurately, acknowledge that

Context:
{context}

Question:
{question}

Answer:
"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        prompt=prompt
    )

def init_embeddings():
    """Initialize the embeddings model."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
