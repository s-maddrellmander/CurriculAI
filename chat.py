import glob
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = "YOUR API KEY"

from langchain.document_loaders import PyPDFDirectoryLoader


def chat():
    pdf_folder_path = os.path.expanduser("~/Zotero/storage/")
    loader = PyPDFDirectoryLoader(
        pdf_folder_path, glob="*.pdf", recursive=True, silent_errors=True
    )
    docs = loader.load()
    print(len(docs))

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="chroma_db/",
        disallowed_special=(),
    )
    vectordb.persist()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory
    )
    query = "Provide a summary of what Graph Neural Networks are used for?"
    result = pdf_qa({"question": query})
    print("Answer:")
    result["answer"]
