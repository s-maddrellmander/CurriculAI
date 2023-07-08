import glob
import logging
import os
from typing import List

from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from tqdm import tqdm

from documents import save_questions_to_file

# os.environ["OPENAI_API_KEY"] = "YOUR API KEY"


logger = logging.getLogger("logger")


def remove_empty_strings(lst):
    """Removes any empty strings from a list"""
    return [item for item in lst if item]


def generate_questions(subject: str, vectordb: Chroma) -> List[str]:
    """Generate a set of functions using Langchain based ona subject

    Args:
        subject (str): The main subject / question stem to generate from

    Returns:
        List[str]: A list of generated questions for later use
    """
    questions = []
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory
    )
    query = (
        subject
        + "\n Generate 25 questions on this subject using the material provided. "
    )
    result = pdf_qa({"question": query})
    print(query)
    print("Answer:")
    print(result["answer"])
    # Split the generated questions into a list of questions
    question_list = result["answer"].split("\n")
    question_list = remove_empty_strings(question_list)
    logger.info(question_list)
    return question_list


def chat(opts, regenerate=False):
    answers = []
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
            vectordb = Chroma(
                persist_directory=chroma_dir, embedding_function=embeddings
            )
        # Generate the relevant questions
        question_list = generate_questions(subject=opts.subject, vectordb=vectordb)

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        pdf_qa = ConversationalRetrievalChain.from_llm(
            OpenAI(temperature=0.8), vectordb.as_retriever(), memory=memory
        )

        for query in tqdm(question_list):
            # query = "Provide a summary of what Graph Neural Networks are used for?"
            result = pdf_qa({"question": query})
            print(query)
            print("Answer:")
            print(result["answer"])
            answers.append(result["answer"])
    logger.info(cb)
    save_questions_to_file(question_list, answers, "question_answers.csv")
