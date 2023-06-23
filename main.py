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


# create logger
logger = logging.getLogger("logging_tryout2")
logger.setLevel(logging.INFO)

OPENAI_API_KEY = "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"


def main():
    with get_openai_callback() as cb:
        # Use the callback to ensure we monitor the usage
        pass


if __name__ == "__main__":
    main()
