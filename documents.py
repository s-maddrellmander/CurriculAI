from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import TokenTextSplitter
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever

import logging
import pickle

from tqdm import tqdm
import os

from agents_chain import get_chains

# Get the same logger by using the same name
logger = logging.getLogger("logger")


def get_docs_for_question_gen(
    text: str, chunk_size: int = 10000, chunk_overlap: int = 1000
):
    # Split text for question generation
    logger.info("Splitting text for question gen")
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts_for_question_gen = text_splitter.split_text(text)

    # Save as documents for further processing
    logger.info("Generating Documents")
    docs_for_question_gen = [Document(page_content=t) for t in texts_for_question_gen]
    return docs_for_question_gen


def get_docs_for_QA(file_path: str):
    # Load Data from PDF for Question Answering
    loader_question_answer = PyPDFLoader(file_path=file_path)
    data_question_answer = loader_question_answer.load()
    return data_question_answer


def load_pdf_pages(file_path: str) -> str:
    loader_question_gen = PdfReader(file_path)
    # Store the text for summarization
    text = ""
    for page in loader_question_gen.pages:
        text += page.extract_text()
    return text


def vector_embeddings(docs_for_vector_database):
    # Check if the vectorstore and faiss index are already created:
    if os.path.exists("database/vectorstore.pkl"):
        with open("database/vectorstore.pkl", "rb") as f:
            db = pickle.load(f)
    else:
        # Create the vector database and reterval QA chain
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        db = FAISS.from_documents(docs_for_vector_database, embeddings)
        # Save the FAISS index and vectorstore
        db.save_local("database/faiss_index")
        with open("database/vectorstore.pkl", "wb") as f:
            pickle.dump(db, f)
    return db
