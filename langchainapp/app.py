import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
import os

# Load API key securely from secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# App UK
st.title("📄 Chat with your PDF or Text File")
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        text = uploaded_file.read().decode("utf-8")

    st.success("✅ File uploaded and text extracted!")

    # Split text for processing
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # Ask a question
    question = st.text_input("Ask a question about the file:")

    if question:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=chunks, question=question)

        st.subheader("📌 Answer:")
        st.write(response)
