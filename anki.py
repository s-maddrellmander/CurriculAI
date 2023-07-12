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

# Refactoring the code into a class


class AnkiCardGenerator:
    def __init__(self, model_name="gpt-3.5-turbo-16k-0613"):
        self.model_name = model_name
        self.logger = logging.getLogger("logger")
        self.template = """
        You are an expert in AI and ML - and you are a helpful AI making insightful 
        and considered Anki style flashcards for a student. The content should be postgraduate level.
        It is important to make the question and answer as detailed as possible. 
        The questions must cover the key concepts but also complex and advanced topics.

        Output format:
        The question and answer should be on the same line, separated by a ; character
        Each card is a new line.

        Generate 15 Anki cards in the above style on the following subject:
        {subject}

        with additional details:
        {details}

        It is really important to use the style described above. 
        ANSWER:
        """

    def _get_chain(self):
        embeddings = OpenAIEmbeddings()

        prompt = PromptTemplate(
            input_variables=["subject", "details"], template=self.template
        )

        llm = ChatOpenAI(model_name=self.model_name)

        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=True,
        )

        return chain

    def _generate(self, subject, details):
        chain = self._get_chain()
        question_list = [subject]
        answers = []
        for query in tqdm(question_list):
            result = chain({"subject": query, "details": details})
            self.logger.info(query)
            self.logger.info("Answer:")
            self.logger.info(result)
            # for line in result["text"].split("\n"):
            #     self.logger.info(line)
        return result["text"]

    def log_result(self, result, verbose):
        if verbose:
            for line in result.split("\n"):
                self.logger.info(line)

    def save_csv(self, data, filename):
        lines = data.split("\n\n")
        pairs = [line.split(";") for line in lines]

        with open(f"{filename}.csv", "w", newline="") as f:
            writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            for pair in pairs:
                writer.writerow([p.strip('"') for p in pair])

    def generate_anki(self, subject, details="", verbose=True):
        with get_openai_callback() as cb:
            result = self._generate(subject, details)
            self.log_result(result, verbose)
            self.save_csv(result, subject.replace(" ", "_"))
            self.logger.info("Generated Anki cards for subject: " + subject)
        self.logger.info(cb)
