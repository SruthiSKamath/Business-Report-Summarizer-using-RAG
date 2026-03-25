import os
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure Gemini
genai.configure(api_key=os.getenv(""))


def load_document(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)

    return loader.load()


def create_rag_pipeline(file_path):
    documents = load_document(file_path)

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # 🔥 FREE embeddings (no API needed)
    embeddings = HuggingFaceEmbeddings()

    # Vector DB
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever()

    return retriever


def summarize(file_path):
    retriever = create_rag_pipeline(file_path)

    # Get relevant chunks
    docs = retriever.get_relevant_documents("Summarize this document")

    context = " ".join([doc.page_content for doc in docs[:5]])

    # 🔥 Gemini LLM
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    You are a business analyst.
    Summarize the following business report clearly:

    {context}
    """

    response = model.generate_content(prompt)

    return response.text
