import csv
import json
import logging
import os
import random
import re
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


class AnkiCardGenerator:
    def __init__(self, model_name="gpt-3.5-turbo-16k-0613"):
        """Initializes the AnkiCardGenerator with the provided model name."""
        self.model_name = model_name
        self.logger = logging.getLogger("logger")
        self.questions_asked = set()

        self.template = """
        You are an expert in AI and ML - and you are a helpful AI making insightful 
        and considered Anki style flashcards for a student. The content should be postgraduate level.
        It is important to make the question and answer as detailed as possible. 
        The questions must cover the key concepts but also complex and advanced topics.

        Generate 15 Anki cards in the above style on the following subject:
        {subject}

        with additional details:
        {details}
        {extra}

        OUTPUT FORMAT:
        The question and answer should be on the same line, separated by a ; character
        Each card is a new line.
        IMPORTANT - DO NOT RESTATE THE QUESTION IN THE ANSWER.
        The questions should make it clear the topic they are asking about.
        
        It is really important to use the style described above. DO NOT NUMBER THE QUESTIONS.
        Question and answer sperated by ; 
        
        ANSWER:
        """

        self.prose_template = """
        You are an AI expert in AI and ML, and your task is to generate a comprehensive, informative, and detailed prose on the given subject. The output should be suitable for a postgraduate level student, covering key concepts as well as complex and advanced topics. Make sure to include examples, explanations, and bullet points for key facts where appropriate.

        Output format:
        The output should be a well-structured prose with clear paragraphs and subheadings. Bullet points can be used to list key facts, important points, or steps in a process. Each major topic or sub-topic should be a new paragraph or section.

        Generate a detailed prose on the following subject:
        {subject}

        with additional details:
        {details}{extra}

        It is really important to use the style described above. 
        ANSWER:
        """

    def __get_chain(self, template):
        """Creates and returns a language model chain using the given template."""
        embeddings = OpenAIEmbeddings()
        prompt = PromptTemplate(
            input_variables=["subject", "details", "extra"], template=template
        )
        llm = ChatOpenAI(model_name=self.model_name)
        return LLMChain(llm=llm, prompt=prompt, verbose=True)

    def __generate_content(self, subject, details, template, extra):
        """Generates content for the given subject and details using the given template."""
        chain = self.__get_chain(template)
        result = chain({"subject": subject, "details": details, "extra": extra})
        self.logger.info(f"Subject: {subject}")
        self.logger.info("Generated Content:")
        return result["text"]

    def __save_csv(self, data, filename):
        """Saves the generated data to a CSV file with the given filename."""
        lines = [
            line for line in data.split("\n") if line.strip()
        ]  # List comprehension to filter out empty lines
        data = "\n".join(lines)  # Join the lines back together
        with open(f"{filename}.csv", "w", newline="") as f:
            f.write(data)
        self.logger.info(f"Saved {len(lines)} lines to the file {filename}.csv.")

    def __save_txt(self, data, filename):
        """Saves the generated data to a text file with the given filename."""
        with open(f"{filename}.txt", "w") as f:
            f.write(data)

    def log_result(self, result, verbose):
        """Logs the generated result if verbose is True."""
        lines = result.split("\n")
        if verbose:
            for line in lines:
                self.logger.info(line)

    def generate(
        self,
        subject,
        details="",
        verbose=True,
        format="anki",
        path="data/",
        extra="",
        save=True,
    ):
        """Generates content for the given subject and details, and saves it to a CSV file."""
        template = self.template if format == "anki" else self.prose_template
        result = self.__generate_content(subject, details, template, extra)
        self.log_result(result, verbose)
        if format == "anki":
            if save is True:
                self.__save_csv(result, os.path.join(path, subject.replace(" ", "_")))
        else:  # format is "prose"
            if save is True:
                self.__save_txt(result, os.path.join(path, subject.replace(" ", "_")))
        self.logger.info(f"Generated content for subject: {subject}")
        return result

    def generate_question(
        self, topic, notes=None, previous_questions=None, aspect=None
    ):
        instruction = f"Generate a unique question about {topic}"
        if aspect:
            instruction += f", focusing on the aspect of {aspect}"
        if previous_questions:
            instruction += ". Avoid duplicating the following questions: " + ", ".join(
                previous_questions
            )
        if notes:
            instruction += f". Consider these notes: {notes}"

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent assistant that's been trained on a diverse range of internet text. You're skilled in generating advanced, postgraduate-level questions.",
                },
                {"role": "user", "content": instruction},
            ],
        )
        new_question = response.choices[0]["message"]["content"].strip()
        if new_question not in self.questions_asked:
            self.questions_asked.add(new_question)
            return new_question
        else:
            return self.generate_question(topic, previous_questions, aspect, notes)

    def generate_answers(self, question_prompt, notes=None, num_options=5):
        instruction = f"For the question: {question_prompt}, generate one correct answer and {num_options - 1} incorrect but plausible answers. The answers should be closely related to the question topic, but clearly incorrect upon close examination."
        if notes:
            instruction += f" Consider these notes: {notes}"

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent assistant that's been trained on a diverse range of internet text. You're skilled in generating advanced, postgraduate-level multiple choice questions.",
                },
                {"role": "user", "content": instruction},
            ],
        )
        raw_answers = response.choices[0]["message"]["content"].strip()
        processed_answers = re.sub(
            r"\b\d+\b", "", raw_answers
        )  # Remove leading numbers
        processed_answers = re.sub(
            r"^[a-zA-Z]\.", "", processed_answers, flags=re.M
        )  # Remove leading letters followed by a dot
        processed_answers = processed_answers.strip().split(
            "\n"
        )  # Split answers by new line
        return [
            answer.strip() for answer in processed_answers if answer.strip()
        ]  # Remove empty answers and extra spaces

    def generate_MCQs(self, topic, notes=None, num_questions=5, num_options=5):
        mcqs = []
        for _ in range(num_questions):
            question_prompt = self.generate_question(topic, notes, self.questions_asked)
            answers = self.generate_answers(question_prompt, notes, num_options)
            correct_answer = answers[0]
            random.shuffle(answers)
            correct_index = answers.index(correct_answer)
            mcqs.append(
                {
                    "question": question_prompt,
                    "answers": answers,
                    "correct_index": correct_index,
                }
            )
        return mcqs, json.dumps(mcqs)

    def combine(self, *inputs, model_name=None):
        # Assume inputs are lists of dictionaries. We join the contents of each input.
        combined_input = " ".join(
            [
                json.dumps(input_dict)
                for input_list in inputs
                for input_dict in input_list
            ]
        )
        if model_name is not None:
            self.model_name = model_name
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an intelligent assistant that's been trained on a diverse range of internet text. You're skilled in generating advanced, postgraduate-level content.",
                },
                {
                    "role": "user",
                    "content": f"I have multiple versions of content here: {combined_input}. Can you generate a single improved version by integrating the best parts of each? Just provde the respone, no preamble.",
                },
            ],
        )

        combined_content = response.choices[0]["message"]["content"].strip()

        return combined_content

    def save_to_file(self, data, path, filename):
        with open(os.path.join(path, filename), "w") as outfile:
            json.dump(data, outfile)
