
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
    """Answer questions using conversation-aware retrieval."""

    chat_history = chat_history or []

    # Step 1: Rewrite follow-up questions into standalone questions.
    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in chat_history[-6:]
    )

    rewrite_prompt = f"""
Convert the current question into a standalone search query
that can be understood without the conversation history.

Use the history only to resolve references such as "it",
"that", or "the second type".
Do not answer the question.
If it is already clear, keep its meaning unchanged.
Return only the standalone query.

Conversation history:
{history_text}

Current question:
{question}

Standalone search query:
"""

    rewritten = llm.invoke(rewrite_prompt)
    search_query = rewritten.content.strip()

    # Step 2: Retrieve chunks using the standalone query.
    results = vector_store.similarity_search(search_query, k=4)

    context = "\n\n".join(
        doc.page_content for doc in results
    )

    # Step 3: Collect source filenames and page numbers.
    sources = []
    for doc in results:
        source_name = Path(
            doc.metadata.get("source", "Unknown source")
        ).name
        page_number = doc.metadata.get("page", 0) + 1
        sources.append(f"{source_name}, page {page_number}")

    # Step 4: Generate an answer grounded in retrieved context.
    prompt = f"""
You are StudyMate, an assistant that explains college notes
in simple, beginner-friendly language.

Answer using the retrieved context.
Use conversation history to understand the discussion, not as
proof of facts.
If the answer is not supported by the context, say that the
information could not be found in the notes.
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
