from unittest.mock import MagicMock, patch

import pytest

import chat
import main
from chat import remove_empty_strings


@pytest.fixture
def mock_opts():
    class Options:
        question_answer = False
        prose_generation = False
        chat = False
        anki = False
        prose = False
        notes = ""
        key = "FAKE API KEY"

    return Options()


@patch("main.generate_questions")
@patch("main.generate_prose")
@patch("main.chat")
def test_main(mock_chat, mock_generate_prose, mock_generate_questions, mock_opts):
    # Test when question_answer is True
    mock_opts.question_answer = True
    main.main(mock_opts)
    mock_generate_questions.assert_called_once_with(mock_opts)

    # Reset mocks and opts
    mock_generate_questions.reset_mock()
    mock_opts.question_answer = False

    # Test when prose_generation is True
    mock_opts.prose_generation = True
    main.main(mock_opts)
    mock_generate_prose.assert_called_once_with(mock_opts)

    # Reset mocks and opts
    mock_generate_prose.reset_mock()
    mock_opts.prose_generation = False

    # Test when chat is True
    mock_opts.chat = True
    main.main(mock_opts)
    mock_chat.assert_called_once()


def test_remove_empty_strings():
    input_list = ["", "", "1. What are Graph Neural Networks?", ""]
    expected_output = ["1. What are Graph Neural Networks?"]
    assert remove_empty_strings(input_list) == expected_output
