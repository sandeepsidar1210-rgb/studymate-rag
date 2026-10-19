
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import FAISS

# Load the API key from the LangChain root .env file
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def build_vector_store(pdf_paths):

    """Load uploaded PDFs and build a searchable FAISS index."""
    documents = []


    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    if not documents:
        raise ValueError("No readable pages were found in the uploaded PDFs.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text could be extracted from these PDFs.")

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # Build FAISS index
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


def answer_question(question, vector_store, llm):
    """Retrieve relevant text and generate an answer."""

    results = vector_store.similarity_search(question, k=3)

    # Build context and source references
    context = "\n\n".join(doc.page_content for doc in results)

    sources = []
    for doc in results:
        source_name = Path(
            doc.metadata.get("source", "Unknown source")
        ).name
        page_number = doc.metadata.get("page", 0) + 1
        sources.append(f"{source_name}, page {page_number}")

    # Ask the LLM to answer from the context
    prompt = f"""
You are StudyMate, an assistant that explains college notes
in simple, beginner-friendly language.

Answer using only the provided context.
If the answer is missing, say it could not be found in the notes.
Do not invent facts.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content, list(dict.fromkeys(sources))
