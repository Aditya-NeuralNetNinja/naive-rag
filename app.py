import os
import fitz  # Ensure PyMuPDF is installed correctly
from langchain.schema import Document  # Import the Document class
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain.prompts import PromptTemplate
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize the chatbot components
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

prompt_template = """
    As a professional assistant, provide a detailed and formally written answer to the question using the provided context. Ensure that the response is professionally formatted and avoids informal language.
    Context:
    {context}
    Question:
    {question}
    Answer:
    """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Function to process the PDF and create a RetrievalQA chain
def process_pdf(uploaded_file):
    # Save the uploaded file to a temporary location
    filepath = "temp.pdf"
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from PDF
    text = ''
    with fitz.open(filepath) as doc:
        for page in doc:
            text += page.get_text()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])  # Pass the text as a list to create_documents

    # Vectorize text chunks
    db = FAISS.from_documents(docs, embeddings)

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever(), prompt=prompt)
    return qa_chain

# Streamlit app
st.title("PDF Chatbot")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    qa_chain = process_pdf(uploaded_file)

    # Query input
    query = st.text_input("Enter your query")

    if query:
        # Get the answer
        answer_dict = qa_chain.invoke(query)

        # Extract the result from the dictionary
        answer = answer_dict.get('result', "No answer found.")

        # Display the answer in a formal manner
        st.markdown(f"**Answer:** {answer.strip()}")