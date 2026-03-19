import streamlit as st
import tempfile
import os
from pdf_qa import load_pdf, create_vectorstore, build_qa_chain, ask_question

st.set_page_config(page_title="PDF Q&A System", page_icon="📄", layout="wide")

st.title("📄 PDF Question Answering System")
st.caption("Powered by Cohere · Upload a PDF → Ask anything about it")  # ← updated

if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.session_state.chain is None:
        with st.spinner("Processing PDF with Cohere... ⏳"):   # ← updated
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_pdf(tmp_path)
            vectorstore = create_vectorstore(chunks)
            st.session_state.chain = build_qa_chain(vectorstore)
            os.unlink(tmp_path)

        st.success(f"✅ Ready! ({len(chunks)} chunks indexed)")

    if st.session_state.chain:
        st.divider()
        if st.button("🔄 Upload New PDF"):
            st.session_state.chain = None
            st.session_state.chat_history = []
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown("""
    **How it works:**
    1. PDF → split into chunks
    2. Chunks → Cohere embeddings
    3. Your question → finds relevant chunks
    4. command-r-plus answers using only those chunks
    """)

if not st.session_state.chain:
    st.info("👈 Upload a PDF from the sidebar to get started!")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "pages" in msg:
                st.caption(f"📍 Sources: Pages {msg['pages']}")

    if question := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                answer, pages = ask_question(
                    st.session_state.chain,
                    question,
                    st.session_state.chat_history
                )
            st.write(answer)
            st.caption(f"📍 Sources: Pages {pages}")

        st.session_state.chat_history.append((question, answer))
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "pages": pages
        })