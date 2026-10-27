import sys
from pathlib import Path
import tempfile
import importlib

# Ensure the StudyMate directory is in sys.path.
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st
import rag
importlib.reload(rag)
from rag import build_vector_store, answer_question
from langchain_google_genai import ChatGoogleGenerativeAI


st.set_page_config(
    page_title="StudyMate",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Initialize session state.
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


st.title("📚 StudyMate")
st.subheader("Your AI College Notes Assistant")
st.write("Ask questions about your college notes.")


uploaded_files = st.file_uploader(
    "Upload your PDF notes",
    type=["pdf"],
    accept_multiple_files=True,
)


@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0,
    )


# Process uploaded PDFs.
if uploaded_files:
    if st.button("Process PDFs", type="primary"):
        try:
            with st.spinner("Processing your PDFs..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    pdf_paths = []

                    for uploaded_file in uploaded_files:
                        safe_name = Path(uploaded_file.name).name
                        pdf_path = Path(temp_dir) / safe_name

                        pdf_path.write_bytes(
                            uploaded_file.getvalue()
                        )

                        pdf_paths.append(pdf_path)

                    vector_store = build_vector_store(pdf_paths)

                    st.session_state.vector_store = vector_store
                    st.session_state.processed_files = [
                        file.name for file in uploaded_files
                    ]

                    # Clear old conversation when new PDFs are processed.
                    st.session_state.messages = []

            st.success("Your PDFs are ready!")

        except Exception as error:
            st.error(f"Could not process the PDFs: {error}")


# Display processed files and previous messages.
if st.session_state.vector_store is not None:
    st.caption(
        "Processed: "
        + ", ".join(st.session_state.processed_files)
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown("### Sources")
                for source_id, source in message["sources"].items():
                    st.caption(
                        f"📄 [{source_id}] {source['filename']} "
                        f"— Page {source['page']}"
                    )


# Accept a new question.
question = st.chat_input("Ask a question about your notes...")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process your PDFs first.")

    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your notes..."):
                try:
                    llm = load_llm()

                    answer, sources = answer_question(
                        question,
                        st.session_state.vector_store,
                        llm,
                        chat_history=st.session_state.messages[:-1],
                    )

                    # Display the answer.
                    st.markdown(answer)

                    if sources:
                        st.markdown("### Sources")
                        for source_id, source in sources.items():
                            st.caption(
                                f"📄 [{source_id}] {source['filename']} "
                                f"— Page {source['page']}"
                            )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })


                except Exception as error:
                    st.error(
                        f"Could not generate an answer: {error}"
                    )
