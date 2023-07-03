import pytest
from unittest.mock import MagicMock, patch
import main  # Make sure to replace this with the correct module where `main` is defined
import chat

@pytest.fixture
def mock_opts():
    class Options:
        question_answer = False
        prose_generation = False
        chat = False
        key = "FAKE API KEY"
    return Options()

@patch('main.generate_questions')
@patch('main.generate_prose')
@patch('chat.chat')
def test_main(mock_chat, mock_generate_prose, mock_generate_questions, mock_opts):
    # Test when question_answer is True
    mock_opts.question_answer = True
    main.main(mock_opts)
    mock_generate_questions.assert_called_once_with(mock_opts)

    # Test when prose_generation is True
    mock_opts.question_answer = False
    mock_opts.prose_generation = True
    main.main(mock_opts)
    mock_generate_prose.assert_called_once_with(mock_opts)

    # Test when chat is True
    mock_opts.prose_generation = False
    mock_opts.chat = True
    main.main(mock_opts)
    mock_chat.assert_called_once()
