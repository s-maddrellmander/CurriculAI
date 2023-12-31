import argparse
import logging
import os

import openai
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import TokenTextSplitter
from tqdm import tqdm

from agents_chain import get_question_answering_chains, get_textbook_chains
from anki import AnkiCardGenerator
from chat import chat
from documents import (
    get_docs_for_QA,
    get_docs_for_question_gen,
    load_pdf_pages,
    save_questions_to_file,
    vector_embeddings,
)

# Create a logger
logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

# Create a console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

# openai.api_key = os.environ["OPENAI_API_KEY"]
# TODO: Need to make this work properly from the env - this is not okay
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# logger.info(OPENAI_API_KEY)
logger.info(openai.api_key)


def main(opts):
    path = "data"
    with get_openai_callback() as cb:
        if opts.question_answer is True:
            generate_questions(opts)
        if opts.prose_generation is True:
            generate_prose(opts)
        if opts.chat is True:
            chat(opts)
        if opts.prose is True:
            generator = AnkiCardGenerator()
            result = generator.generate(opts.subject, opts.notes, format="prose")
        if opts.anki is True:
            generator = AnkiCardGenerator()
            result = generator.generate(
                opts.subject, opts.notes, format="anki", extra="", verbose=False
            )
        if opts.mcq is True:
            generator = AnkiCardGenerator(model_name="gpt-3.5-turbo")
            mcq, json_mcq = generator.generate_MCQs(opts.subject, opts.notes)
            for mc in mcq:
                print(mc)
            generator.save_to_file(
                mcq, path, f"{opts.subject.replace(' ', '_')}_json.txt"
            )

    logger.info(cb)


def generate_prose(opts):
    with get_openai_callback() as cb:
        # Use the callback to ensure we monitor the usage
        file_path = "thebook.pdf"

        # Parse the pdf to text
        text = load_pdf_pages(file_path)
        # docs_for_question_gen = get_docs_for_question_gen(text)
        # logger.info(
        #     f"Summary: {len(docs_for_question_gen[0].page_content)} string length of first element"
        # )

        # Load Data from PDF for Question Answering
        data_question_answer = get_docs_for_QA(file_path=file_path)
        logger.info(
            f"Summary: {len(data_question_answer[0].page_content)} document length of first element"
        )

        # Split data for question answering vector database
        logger.info("Splitting text for answering database")
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs_for_vector_database = text_splitter.split_documents(data_question_answer)

        # Get the vector embeddings
        db = vector_embeddings(docs_for_vector_database=docs_for_vector_database)

        # Get the LLM chains
        # llm_question_answer, question_chain = get_question_answering_chains(
        #     OPENAI_API_KEY=OPENAI_API_KEY
        # )
        llm_question_answer, question_chain = get_textbook_chains(
            OPENAI_API_KEY=openai.api_key,
            textbook_section=opts.subject,
        )

        # Run the question generation chain
        # TODO: How is it best to control the number of docs here?
        # The "questions" when we want a single summary is not a good approach
        retriever = db.as_retriever()
        docs = retriever.get_relevant_documents(
            opts.subject,
        )
        # question = question_chain.run(docs)
        # SPlit the generated questions into a list of questions
        # question_list = questions.split("\n")
        # import pdb; pdb.set_trace()

        # Answer each question
        # for question in tqdm(question_list):
        prompt_template_initial = """

            Your goal is to prepare a student for their exam in Machine Learning and AI.
            You are an expert in the field of Machine Learning and AI and you are writing a textbook.
            The section of the textbook to be written is titled:
            {textbook_section}
            This is the topic this section needs to be about. 
            Focus on this specific topic in the answer. 
            
            You do this by writing a detailed page of an advanced level textbook using the following text:

            {text}

            Think step by step.

            We are writing an advanced level textbook - treating the students with respect,
            but having high expectations of their ability. 
            This is a textbook targeting postgraduates, advanced level content. 

            Exam criteria:

            * Complete and deep understanding of the results 
            * Sophisticated understanding of machine learning
            * World leading expert in AI
            
            

            Make sure not to lose any important information. Be as detailed as possible. 
            Long form answer - make sure everything is explained in detail. 
            """

        # refine_template_prompt = """

        # Your goal is to prepare a student for their exam in Machine Learning and AI.
        # You are an expert in the field of Machine Learning and AI.

        # We have recieved an initial draft of the section for the textbook: {existing_answer}.
        # We have the option to refine the existing text or completely update.
        # (Only if necessary) with some more context below
        # "------------\n"
        # "{text}\n"
        # "------------\n"

        # Given the new context, refine the original textbook section, remeber the section of the textbook to be written is titled:
        # {textbook_section}

        # * Complete and deep understanding of the results
        # * Sophisticated understanding of machine learning
        # * World leading expert in AI
        # """
        from langchain.prompts import PromptTemplate

        textbook_section = opts.subject
        PROMPT_QUESTIONS = PromptTemplate(
            template=prompt_template_initial,
            input_variables=["text"],
            partial_variables={"textbook_section": textbook_section},
        )
        # NOTE: This is not really appropriate for general prose generation.
        # Create the question generation chain
        qa = load_summarize_chain(
            llm=llm_question_answer,
            chain_type="stuff",
            verbose=True,
            prompt=PROMPT_QUESTIONS,
        )

        answer = qa.run(docs)
        print("Answer: ", answer)
        print("------------------------------------------------------------\n\n")

        print(cb)


def generate_questions(opts):
    with get_openai_callback() as cb:
        # Use the callback to ensure we monitor the usage
        file_path = "thebook.pdf"
        logger.info(f"Openai API Key openai.api_key: {openai.api_key}")

        # Parse the pdf to text
        text = load_pdf_pages(file_path)
        docs_for_question_gen = get_docs_for_question_gen(text)
        logger.info(
            f"Summary: {len(docs_for_question_gen[0].page_content)} string length of first element"
        )

        # Load Data from PDF for Question Answering
        data_question_answer = get_docs_for_QA(file_path=file_path)
        logger.info(
            f"Summary: {len(data_question_answer[0].page_content)} document length of first element"
        )

        # Split data for question answering vector database
        logger.info("Splitting text for answering database")
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs_for_vector_database = text_splitter.split_documents(data_question_answer)

        # Get the vector embeddings
        db = vector_embeddings(docs_for_vector_database=docs_for_vector_database)

        # Get the LLM chains
        llm_question_answer, question_chain = get_question_answering_chains(
            OPENAI_API_KEY=openai.api_key, textbook_section=opts.subject
        )
        # llm_question_answer, question_chain = get_textbook_chains(
        #     OPENAI_API_KEY=OPENAI_API_KEY,
        #     textbook_section="3.2 Supervised Learning Methods",
        # )

        # NOTE: This is not really appropriate for general prose generation.
        # import pdb; pdb.set_trace()
        qa = RetrievalQA.from_chain_type(
            llm=llm_question_answer, chain_type="stuff", retriever=db.as_retriever()
        )

        # Run the question generation chain
        # TODO: How is it best to control the number of docs here?
        # The "questions" when we want a single summary is not a good approach
        questions = question_chain.run(docs_for_question_gen)
        # SPlit the generated questions into a list of questions
        question_list = questions.split("\n")

        # Answer each question
        answers = []
        for question in tqdm(question_list):
            print("Question: ", question)
            answer = qa.run(question)
            answers.append(answer)
            print("Answer: ", answer)
            print("------------------------------------------------------------\n\n")

        print(cb)
    if opts.save_questions:
        save_questions_to_file(questions, answers, "question_answers.csv")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program.")
    parser.add_argument("--key", help="Your OpenAI API key", default=None)
    parser.add_argument(
        "--question_answer",
        help="Generate questions and answers",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--prose_generation",
        help="Generate prose ont he topic",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--subject",
        help="Topic heading for generation",
        default="Introduction to ML.",
        type=str,
    )
    parser.add_argument(
        "--chat",
        help="Use the model in the chat mode",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save_questions",
        help="Save the questions to an anki compatible csv file",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--anki",
        help="Boolean to generate simple anki cards",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--prose",
        help="Boolean to generate simple prose cards",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--notes",
        help="Add more free text details to guide the questions",
        default="",
        action="store",
    )
    parser.add_argument(
        "--mcq",
        help="Generate MCQ questions",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    logger.info(args)
    main(args)
