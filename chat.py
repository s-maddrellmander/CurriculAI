import glob
import os

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.callbacks import get_openai_callback
# os.environ["OPENAI_API_KEY"] = "YOUR API KEY"

from langchain.document_loaders import PyPDFDirectoryLoader
import logging

logger = logging.getLogger("logger")


def chat(opts, regenerate = False):
    with get_openai_callback() as cb:
        embeddings = OpenAIEmbeddings()

        # Check for existing vectored database and regenerate if necessary
        chroma_dir = "chroma_db/"
        if regenerate or not os.path.exists(chroma_dir):
            pdf_folder_path = os.path.expanduser("~/Zotero/storage/")
            loader = PyPDFDirectoryLoader(
                pdf_folder_path, glob="*.pdf", recursive=True, silent_errors=True
            )
            docs = loader.load()
            print(len(docs))

            vectordb = Chroma.from_documents(
                docs,
                embedding=embeddings,
                persist_directory=chroma_dir,
                disallowed_special=(),
            )
            vectordb.persist()
        else:
            vectordb = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        pdf_qa = ConversationalRetrievalChain.from_llm(
            OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory
        )
        query = "Provide a summary of what Graph Neural Networks are used for?"
        result = pdf_qa({"question": query})
        print(query)
        print("Answer:")
        print(result["answer"])
    logger.info(cb)
