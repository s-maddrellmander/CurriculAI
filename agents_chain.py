import logging
import pickle

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


def get_chains(OPENAI_API_KEY):
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

    # Create the LLM model or the questions answering
    llm_question_answer = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, temperature=0.4, model="gpt-3.5-turbo"
    )

    return llm_question_answer, question_chain
