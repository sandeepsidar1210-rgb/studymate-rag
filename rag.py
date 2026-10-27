import re
from pathlib import Path
from dotenv import load_dotenv

def extract_text(content):
    """Safely extract string text from model response content (str, list of dicts, or other)."""
    if isinstance(content, list):
        return "\n".join(
            item["text"] if isinstance(item, dict) and "text" in item
            else str(item)
            for item in content
        ).strip()
    return str(content).strip()

def filter_and_format_citations(answer, source_map):
    """
    Validate and preserve model-generated inline citations against retrieved sources.
    - Preserves valid citations (e.g. [Source 1] or [Source 1, Source 2]).
    - Filters out invalid source IDs that were not retrieved.
    - If the answer indicates information was not found in the notes, attaches no sources.
    - Returns the cleaned answer and a dictionary of only the sources actually cited.
    """
    not_found_patterns = [
        r"could not be found",
        r"cannot be found",
        r"can not be found",
        r"not found in (the )?(uploaded )?notes",
        r"not mentioned in (the )?(uploaded )?notes",
        r"could not find",
        r"cannot find",
        r"no information .* found",
        r"not contain .* information",
    ]
    lower_ans = answer.lower()
    is_not_found = any(re.search(pat, lower_ans) for pat in not_found_patterns)

    cited_ids = []

    def replace_bracket(match):
        content = match.group(1)
        nums = re.findall(r"\d+", content)
        valid_nums = [n for n in nums if f"Source {n}" in source_map]

        if is_not_found or not valid_nums:
            return ""

        for n in valid_nums:
            s_id = f"Source {n}"
            if s_id not in cited_ids:
                cited_ids.append(s_id)

        return "[" + ", ".join(f"Source {n}" for n in valid_nums) + "]"

    cleaned_answer = re.sub(
        r"\[([^\]]*Source[^\]]*)\]", replace_bracket, answer, flags=re.IGNORECASE
    )
    cleaned_answer = re.sub(r"\s+([.,!?;])", r"\1", cleaned_answer)
    cleaned_answer = re.sub(r" {2,}", " ", cleaned_answer).strip()

    if is_not_found:
        cited_sources = {}
    else:
        cited_sources = {
            s_id: source_map[s_id] for s_id in cited_ids if s_id in source_map
        }

    return cleaned_answer, cited_sources

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
    search_query = extract_text(rewritten.content)
    if not search_query:
        search_query = question

    # Step 2: Retrieve chunks using the standalone query.
    results = vector_store.similarity_search(search_query, k=4)



    # Step 3: Collect source filenames and page numbers.
    # sources = []
    # for doc in results:
    #     source_name = Path(
    #         doc.metadata.get("source", "Unknown source")
    #     ).name
    #     page_number = doc.metadata.get("page", 0) + 1
    #     sources.append(f"{source_name}, page {page_number}")

    
    # Build context with reliable source IDs.
    context_parts = []
    source_map = {}

    for i, doc in enumerate(results, start=1):
        source_id = f"Source {i}"

        source_name = Path(
            doc.metadata.get("source", "Unknown source")
        ).name

        # PyPDFLoader page numbers are normally zero-indexed.
        page_number = doc.metadata.get("page", 0) + 1

        source_map[source_id] = {
            "filename": source_name,
            "page": page_number,
        }

        context_parts.append(
            f"[{source_id}]\n"
            f"Filename: {source_name}\n"
            f"Page: {page_number}\n"
            f"Passage: {doc.page_content}"
        )

    context = "\n\n".join(context_parts)


    # Step 4: Generate an answer grounded in retrieved context.
    prompt = f"""
    You are StudyMate, an AI assistant that explains college
    notes in simple, beginner-friendly language.

    Instructions:
    1. Read all retrieved passages carefully before answering.
    2. Combine relevant information from different passages when needed.
    3. Explain concepts clearly using simple, beginner-friendly words.
    4. Use only information supported by the retrieved passages.
    5. If the passages contain related information but not an exact
       definition, explain only what can reasonably be concluded.
    6. If the passages do not contain enough useful information,
       say that the answer could not be found in the uploaded notes.
    7. Use conversation history to understand follow-up questions,
    but do not treat previous AI answers as evidence.
    8. Cite every factual claim based on the retrieved passages
        using the exact source ID provided, such as [Source 1].
    9. Put the citation immediately after the claim it supports.
    10. Never invent source IDs, filenames, or page numbers.
    11. If no retrieved passage supports the answer, say that
        the answer could not be found in the uploaded notes.
    12. Do not include a separate bibliography; citations will be
        mapped to PDF filenames and page numbers by the application.
    


    Conversation history:
    {history_text}

    Retrieved context:
    {context}

    Current question:
    {question}

    Answer:
    """

    
    
    
    response = llm.invoke(prompt)
    raw_answer = extract_text(response.content)

    final_answer, cited_sources = filter_and_format_citations(raw_answer, source_map)
    return final_answer, cited_sources
