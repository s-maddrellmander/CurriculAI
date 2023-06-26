import logging
import os

from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.text_splitter import TokenTextSplitter
from tqdm import tqdm

from agents_chain import get_question_answering_chains, get_textbook_chains
from documents import (
    get_docs_for_QA,
    get_docs_for_question_gen,
    load_pdf_pages,
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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    with get_openai_callback() as cb:
        # Use the callback to ensure we monitor the usage
        file_path = "thebook.pdf"

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
            OPENAI_API_KEY=OPENAI_API_KEY
        )
        # llm_question_answer, question_chain = get_textbook_chains(
        #     OPENAI_API_KEY=OPENAI_API_KEY,
        #     textbook_section="3.2 Supervised Learning Methods",
        # )

        # NOTE: This is not really appropriate for general prose generation.
        qa = RetrievalQA.from_chain_type(
            llm=llm_question_answer, chain_type="stuff", retriever=db.as_retriever()
        )

        # Run the question generation chain
        # TODO: How is it best to control the number of docs here?
        # The "questions" when we want a single summary is not a good approach
        questions = question_chain.run(docs_for_question_gen)
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


if __name__ == "__main__":
    main()
