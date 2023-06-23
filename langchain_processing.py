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


# """
# TODO:

# 1. How to control the number of questions generated?
# 2. Use to generate multiple choice questions
# 3. Save the tokenised vector db - persist - to use again later

# """


with get_openai_callback() as cb:
    file_path = "the-odyssey.pdf"
    loader_question_gen = PdfReader(file_path)

    # Store the text for summarization
    text = ""
    for page in loader_question_gen.pages[:50]:
        text += page.extract_text()

    # Split text for question generation
    logger.info("Splitting text for question gen")
    text_splitter = TokenTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )  # model_name="gpt-3.5-turbo-16k",
    texts_for_question_gen = text_splitter.split_text(text)

    # Save as documents for further processing
    logger.info("Gnerating Documents")
    docs_for_question_gen = [Document(page_content=t) for t in texts_for_question_gen]

    # Load Data from PDF for Question Answering
    loader_question_answer = PyPDFLoader(file_path=file_path)
    data_question_answer = loader_question_answer.load()

    # Split data for question answering vector database
    logger.info("Splitting text for answering database")
    text_splitter = TokenTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )  # model_name="gpt-3.5-turbo-16k",
    docs_for_vector_database = text_splitter.split_documents(data_question_answer)

    prompt_template_questions = """

    Your goal is to prepare a student for their exam in Machine Learning and AI.
    You are an expert in the field of Machine Learning and AI.
    You do this by asking questions about the text below:

    {text}

    Think step by step.

    Create questions that will prepare the student for their exam.

    Exam criteria:

    * Complete and deep understanding of the results 
    * Sophisticated understanding of machine learning
    * World leading expert in AI
    * We need 25 questions

    Make sure not to lose any important information. Be as detailed as possible. 
    Create questions that will prepare the student for the exam.
    QUESTIONS IN YOUR PREFERRED LANGUAGE:

    """

    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template_questions, input_variables=["text"]
    )

    refine_template_questions = """

    Your goal is to prepare a student for their exam in Machine Learning and AI.
    You are an expert in the field of Machine Learning and AI.

    We have recieved some questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (Only if necessary) with some more context below
    "------------\n"
    "{text}\n"
    "------------\n"

    Given the new context, refine the original questions in YOUR LANGUAGE.
    Create questions that will prepare the student for their exam.

    * Complete and deep understanding of the results 
    * Sophisticated understanding of machine learning
    * World leading expert in AI
    * We need 25 questions
    """

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template_questions,
    )

    # Create the LLM model for the questions generation
    llm_question_gen = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, temperature=0.4, model="gpt-3.5-turbo-16k"
    )

    # Ceate the question generation chain
    question_chain = load_summarize_chain(
        llm=llm_question_gen,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

    # Run the question generation chain
    questions = question_chain.run(docs_for_question_gen)

    # Create the LLM model or the questions answering
    llm_question_answer = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, temperature=0.4, model="gpt-3.5-turbo"
    )

    # Create the vector database and reterval QA chain
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(docs_for_vector_database, embeddings)
    # You can also save and load a FAISS index. This is useful so you don’t have to recreate it everytime you use it.
    db.save_local("database/faiss_index")
    with open("database/vectorstore.pkl", "wb") as f:
        pickle.dump(db, f)
    # new_db = FAISS.load_local("faiss_index", embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=llm_question_answer, chain_type="stuff", retriever=db.as_retriever()
    )

    # SPlit the generated questions into a list of questions
    question_list = questions.split("\n")
    # import ipdb; ipdb.set_trace()

    # Answer each question
    for question in tqdm(question_list):
        print("Question: ", question)
        answer = qa.run(question)
        print("Answer: ", answer)
        print("------------------------------------------------------------\n\n")

    print(cb)
