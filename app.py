import streamlit as st

from findocs_qa.generation.answer import Answerer

st.set_page_config(page_title="FinDocs QA", page_icon="📊", layout="centered")


@st.cache_resource
def get_answerer() -> Answerer:
    """Load the models once and reuse them for every question."""
    return Answerer()


st.title("📊 FinDocs QA")
st.caption(
    "Ask questions about the FY2024-25 annual reports of HDFC Bank, Infosys and Lenskart. "
    "Every answer cites the page it came from."
)

question = st.text_input(
    "Your question",
    placeholder="e.g. How many employees does HDFC Bank have?",
)

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Searching the reports..."):
        result = get_answerer().answer(question)

    st.subheader("Answer")
    st.markdown(result["answer"])

    st.subheader("Sources")
    for c in result["chunks"]:
        with st.expander(f"{c['doc_id']} · page {c['page']}"):
            st.text(c["text"])

    st.caption(f"Tokens used: {result['tokens']}")