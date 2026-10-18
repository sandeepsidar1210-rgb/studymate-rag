import sys
from pathlib import Path

# Ensure the studymate directory is in sys.path so rag.py can be found regardless of current working directory
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import streamlit as st
from rag import build_vector_store, answer_question  # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI

st.set_page_config(
    page_title="StudyMate",
    page_icon="📚",
)

st.title("📚 StudyMate")
st.subheader("Your AI College Notes Assistant")
st.write("Ask questions about your college notes.")

@st.cache_resource
def load_resources():
    vector_store = build_vector_store()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )

    return vector_store, llm

try:
    with st.spinner("Preparing your notes..."):
        vector_store, llm = load_resources()

    question = st.text_input(
        "Ask a question",
        placeholder="Explain CPU scheduling...",
    )

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching your notes..."):
                answer, sources = answer_question(
                    question,
                    vector_store,
                    llm,
                )

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Sources")
            for source in sources:
                st.write(f"- {source}")

except Exception as error:
    st.error(f"Something went wrong: {error}")
