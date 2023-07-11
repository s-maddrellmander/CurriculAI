import csv
import logging
import os
import textwrap
from getpass import getpass

import chromadb
import langchain
import openai
from langchain.callbacks import get_openai_callback
from langchain.chains import (
    ConversationalRetrievalChain,
    ConversationChain,
    LLMBashChain,
    LLMChain,
    RetrievalQA,
    SimpleSequentialChain,
)
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

from documents import save_questions_to_file

# os.environ["OPENAI_API_KEY"] = "YOUR API KEY"


logger = logging.getLogger("logger")


def save_csv(data: str, filename: str):
    # Split the data into lines, then split each line into question-answer pairs
    lines = data.split("\n\n")
    pairs = [line.split(";") for line in lines]

    # Write to CSV
    with open(f"data/{filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(['Question', 'Answer'])  # Write header
        for pair in pairs:
            writer.writerow([p.strip('"') for p in pair])  # Write data


def generate_anki(subject: str) -> str:
    """Simple main function to generate anki cards on a subject

    Args:
        subject (str): string for question generation
    """

    with get_openai_callback() as cb:
        embeddings = OpenAIEmbeddings()

        template = """
        You are an expert in AI and ML - and you are a helpful AI making insightful 
        and considered Anki style flashcards for a student. The content should be postgraduate level.
        It is important to make the question and answer as detailed as possible. 
        The questions must cover the key concepts but also complex and advanced topics.
        
        Output format:
        The question and answer should be on the same line, separated by a ; character
        Each card is a new line.
        

        Generate 25 Anki cards in the above style on the following subject:
        {subject}

        It is really important to use the style described above. 
        ANSWER:
        """
        prompt = PromptTemplate(input_variables=["subject"], template=template)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613")

        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
        )

        question_list = [subject]
        answers = []
        for query in tqdm(question_list):
            # query = "Provide a summary of what Graph Neural Networks are used for?"
            result = chain({"subject": query})
            print(query)
            print("Answer:")
            print(result)
            for line in result["text"].split("\n"):
                print(line)
    print(subject)
    save_csv(result["text"], subject.replace(" ", "_"))

    logger.info(cb)
