import os
import fitz  # PyMuPDF for PDF processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import streamlit as st
from docx import Document

# Load environment variables
load_dotenv()

# Initialize the chatbot components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

# Define the prompt
prompt_template = """
As a professional assistant, provide a detailed and formally written answer to the question using the provided context. Ensure that the response is professionally formatted and avoids informal language.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ''
    with fitz.open(pdf_file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract text from Word file
def extract_text_from_docx(word_file_path):
    doc = Document(word_file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

# Process uploaded files
def process_files(pdf_text, word_text):
    # Combine texts
    text = pdf_text + '\n' + word_text

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # Vectorize text chunks
    db = FAISS.from_documents(docs, embeddings)

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)
    return qa_chain

# Streamlit app
st.title("PDF & Word Doc Chatbot")

# File uploader
uploaded_files = st.file_uploader("Upload PDF and/or Word file(s)", type=["docx", "pdf"], accept_multiple_files=True)

if uploaded_files:
    pdf_text = ""
    word_text = ""

    # Process uploaded files
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".pdf"):
            pdf_text += extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            word_text += extract_text_from_docx(uploaded_file)

    if pdf_text or word_text:
        qa_chain = process_files(pdf_text, word_text)

        # Query input
        query = st.text_input("Enter your query")

        if query:
            # Get the answer
            result = qa_chain.invoke(query)
            answer = result["result"]
            st.write(f"**Answer:** {answer}")
    else:
        st.error("No text extracted from the uploaded files.")
