import logging
import pickle

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


def get_prompt_templates():
    # All the template creation code goes here
    prompt_template_initial = """

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
    """

    refine_template_prompt = """

    Your goal is to prepare a student for their exam in Machine Learning and AI.
    You are an expert in the field of Machine Learning and AI.

    We have recieved some questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (Only if necessary) with some more context below
    "------------\n"
    "{text}\n"
    "------------\n"

    Given the new context, refine the original questions.
    Create questions that will prepare the student for their exam.

    * Complete and deep understanding of the results 
    * Sophisticated understanding of machine learning
    * World leading expert in AI
    * We need 25 questions
    """

    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template_initial,
        input_variables=["text"],
    )
    # PROMPT_QUESTIONS.partial(textbook_section=textbook_section)
    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template_prompt,
    )

    return PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS


def create_llm_model(openai_api_key: str, temperature: float, model: str):
    return ChatOpenAI(
        openai_api_key=openai_api_key, temperature=temperature, model=model
    )


def get_question_answering_chains(OPENAI_API_KEY: str):
    PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS = get_prompt_templates()

    # Create the LLM model for the questions generation
    llm_question_gen = create_llm_model(OPENAI_API_KEY, 0.4, "gpt-3.5-turbo-16k")

    # Create the question generation chain
    question_chain = load_summarize_chain(
        llm=llm_question_gen,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

    # Create the LLM model or the questions answering
    llm_question_answer = create_llm_model(OPENAI_API_KEY, 0.4, "gpt-3.5-turbo")

    return llm_question_answer, question_chain


def get_textbook_chains(OPENAI_API_KEY: str, textbook_section: str):
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

    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template_initial,
        input_variables=["text"],
        partial_variables={"textbook_section": textbook_section},
    )
    # PROMPT_QUESTIONS.partial(textbook_section=textbook_section)
    # REFINE_PROMPT_QUESTIONS = PromptTemplate(
    #     input_variables=["existing_answer", "text"],
    #     partial_variables={"textbook_section": textbook_section},
    #     template=refine_template_prompt,
    # )

    # REFINE_PROMPT_QUESTIONS.partial(textbook_section=textbook_section)
    # Create the LLM model for the questions generation
    llm_question_gen = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, temperature=0.4, model="gpt-3.5-turbo-16k"
    )

    # Ceate the question generation chain
    question_chain = load_summarize_chain(
        llm=llm_question_gen,
        chain_type="stuff",
        verbose=True,
        prompt=PROMPT_QUESTIONS,
        # refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

    # Create the LLM model or the questions answering
    llm_question_answer = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY, temperature=0.4, model="gpt-3.5-turbo"
    )

    return llm_question_answer, question_chain
