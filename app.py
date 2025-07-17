import streamlit as st
from document_processor import chunk_and_embed
from main import get_qa_chain

st.set_page_config(page_title="📄 Document Q&A", layout="wide")

st.title("📄 Ask Your Documents Anything")

if st.button("🔄 Process Documents"):
    with st.spinner("Processing..."):
        chunk_and_embed()
    st.success("Documents processed and embedded!")

question = st.text_input("❓ Ask a question about the documents:")
if question:
    qa_chain = get_qa_chain()
    with st.spinner("Thinking..."):
        result = qa_chain({"query": question})
        st.write("### Answer:")
        st.write(result["result"])

        with st.expander("🔍 Source Documents"):
            for doc in result["source_documents"]:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.text(doc.page_content[:1000])  # Preview first 1000 chars
