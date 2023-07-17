import csv
import logging
import os
from unittest.mock import MagicMock, patch

from anki import AnkiCardGenerator


def test_anki_card_generator_creation():
    generator = AnkiCardGenerator()
    assert isinstance(generator, AnkiCardGenerator)


def test_save_csv():
    generator = AnkiCardGenerator()
    test_data = "Question;Answer"
    test_filename = "test_file"
    generator._AnkiCardGenerator__save_csv(test_data, test_filename)
    with open(f"{test_filename}.csv", "r") as file:
        content = file.read()
    assert content == test_data
    os.remove(f"{test_filename}.csv")  # clean up the test file


def test_save_txt():
    generator = AnkiCardGenerator()
    test_data = "This is a test.\n"
    test_filename = "test_file"
    generator._AnkiCardGenerator__save_txt(test_data, test_filename)
    with open(f"{test_filename}.txt", "r") as file:
        content = file.read()
    assert content == test_data
    os.remove(f"{test_filename}.txt")  # clean up the test file


def test_logging():
    with patch("logging.Logger.info") as mock_info, patch.object(
        AnkiCardGenerator, "_AnkiCardGenerator__get_chain"
    ) as mock_get_chain:
        # Create a mock chain that returns a predictable result
        mock_chain = MagicMock()
        mock_chain.return_value = {"text": "test_result"}
        mock_get_chain.return_value = mock_chain

        generator = AnkiCardGenerator()
        result = generator._AnkiCardGenerator__generate_content(
            "test_subject", "details", "extra", generator.template
        )
        assert mock_info.call_count == 2  # check if info was called 3 times
        assert result == "test_result"


def test_log_result():
    with patch("logging.Logger.info") as mock_info:
        generator = AnkiCardGenerator()
        result = "Line 1\nLine 2\nLine 3"
        generator.log_result(result, verbose=True)
        assert mock_info.call_count == 3  # check if info was called 3 times


def test_generate():
    with patch.object(
        AnkiCardGenerator, "_AnkiCardGenerator__generate_content"
    ) as mock_generate_content, patch.object(
        AnkiCardGenerator, "_AnkiCardGenerator__save_csv"
    ) as mock_save_csv:
        # Make __generate_content return a predictable result
        mock_generate_content.return_value = "test_result"

        generator = AnkiCardGenerator()
        result = generator.generate(
            "test_subject", "details", format="anki", extra="extra"
        )
        # Check that __generate_content and __save_csv were called with the correct arguments
        mock_generate_content.assert_called_with(
            "test_subject", "details", generator.template, "extra"
        )
        mock_save_csv.assert_called_with("test_result", "data/test_subject")

    assert result == "test_result"


import json

from anki import AnkiCardGenerator


def test_generate_question():
    gen = AnkiCardGenerator()
    question = gen.generate_question("Machine Learning")
    assert isinstance(question, str)
    assert question  # check that the question is not empty


def test_generate_answers():
    gen = AnkiCardGenerator()
    answers = gen.generate_answers("What is machine learning?", 5)
    assert isinstance(answers, list)
    # assert len(answers) == 5  # check that the correct number of answers are generated
    for answer in answers:
        assert isinstance(answer, str)
        # assert answer  # check that the answer is not empty


def test_generate_MCQs():
    gen = AnkiCardGenerator(
        "gpt-3.5-turbo"
    )  # Instantiate your class with the appropriate model name
    mcqs, _ = gen.generate_MCQs(
        "Machine Learning", 5, 5
    )  # get the python object, not the json string
    assert isinstance(mcqs, list)
    # assert len(mcqs) == 5  # check that the correct number of MCQs are generated
    for mcq in mcqs:
        assert set(mcq.keys()) == {
            "question",
            "answers",
            "correct_index",
        }  # check that the MCQ has the correct structure
        # assert (
        #     len(mcq["answers"]) == 5
        # )  # check that the correct number of answers are generated
