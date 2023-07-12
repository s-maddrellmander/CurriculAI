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
        """Initializes the AnkiCardGenerator with the provided model name."""
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

        self.prose_template = """
        You are an AI expert in AI and ML, and your task is to generate a comprehensive, informative, and detailed prose on the given subject. The output should be suitable for a postgraduate level student, covering key concepts as well as complex and advanced topics. Make sure to include examples, explanations, and bullet points for key facts where appropriate.

        Output format:
        The output should be a well-structured prose with clear paragraphs and subheadings. Bullet points can be used to list key facts, important points, or steps in a process. Each major topic or sub-topic should be a new paragraph or section.

        Generate a detailed prose on the following subject:
        {subject}

        with additional details:
        {details}

        It is really important to use the style described above. 
        ANSWER:
        """

    def __get_chain(self, template):
        """Creates and returns a language model chain using the given template."""
        embeddings = OpenAIEmbeddings()
        prompt = PromptTemplate(
            input_variables=["subject", "details"], template=template
        )
        llm = ChatOpenAI(model_name=self.model_name)
        return LLMChain(llm=llm, prompt=prompt, verbose=True)

    def __generate_content(self, subject, details, template):
        """Generates content for the given subject and details using the given template."""
        chain = self.__get_chain(template)
        result = chain({"subject": subject, "details": details})
        self.logger.info(f"Subject: {subject}")
        self.logger.info("Generated Content:")
        self.logger.info(result)
        return result["text"]

    def __save_csv(self, data, filename):
        """Saves the generated data to a CSV file with the given filename."""
        with open(f"{filename}.csv", "w", newline="") as f:
            f.write(data)

    def log_result(self, result, verbose):
        """Logs the generated result if verbose is True."""
        if verbose:
            for line in result.split("\n"):
                self.logger.info(line)

    def generate(self, subject, details="", verbose=True, format="anki", path="data/"):
        """Generates content for the given subject and details, and saves it to a CSV file."""
        template = self.template if format == "anki" else self.prose_template
        result = self.__generate_content(subject, details, template)
        self.log_result(result, verbose)
        self.__save_csv(result, os.path.join(path, subject.replace(" ", "_")))
        self.logger.info(f"Generated content for subject: {subject}")
