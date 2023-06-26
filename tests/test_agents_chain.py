import pytest
from agents_chain import get_prompt_templates, create_llm_model
import os
from langchain.prompts import PromptTemplate


def test_get_prompt_templates():
    PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS = get_prompt_templates(textbook_section="fake topic")

    assert isinstance(PROMPT_QUESTIONS, PromptTemplate)
    assert isinstance(REFINE_PROMPT_QUESTIONS, PromptTemplate)

    # Here you could add more asserts to check the content of the templates...


@pytest.mark.parametrize(
    "temperature, model", [(0.4, "gpt-3.5-turbo"), (0.7, "gpt-3.5-turbo-16k")]
)
def test_create_llm_model(temperature, model):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = create_llm_model(openai_api_key, temperature, model)

    assert llm.openai_api_key == openai_api_key
    assert llm.temperature == temperature
    assert llm.model_name == model

    # Here you could add more asserts depending on your specific needs...
