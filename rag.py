
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



def answer_question(question, vector_store, llm, chat_history=None):
    """Answer a question using retrieved notes and conversation history."""

    chat_history = chat_history or []

    # Retrieve relevant chunks from the uploaded PDFs.
    results = vector_store.similarity_search(question, k=4)

    context = "\n\n".join(
        doc.page_content for doc in results
    )

    sources = []
    for doc in results:
        source_name = Path(
            doc.metadata.get("source", "Unknown source")
        ).name
        page_number = doc.metadata.get("page", 0) + 1
        sources.append(f"{source_name}, page {page_number}")

    # Include recent conversation turns for follow-up questions.
    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history[-6:]
    )

    prompt = f"""
    You are StudyMate, an assistant that explains college notes
    in simple, beginner-friendly language.

    Use the retrieved context as the source of factual answers.
    Use conversation history only to understand follow-up questions.
    Do not treat earlier assistant answers as proof of facts.
    If the answer cannot be supported by the retrieved context,
    say that the information could not be found in the notes.
    Do not invent facts.

    Conversation history:
    {history_text}

    Retrieved context:
    {context}

    Current question:
    {question}

    Answer:
    """

    response = llm.invoke(prompt)

    return response.content, list(dict.fromkeys(sources))
