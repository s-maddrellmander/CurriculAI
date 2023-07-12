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
    generator.save_csv(test_data, test_filename)
    with open(f"{test_filename}.csv", "r") as file:
        content = file.read().strip()  # remove trailing newline
    assert content == test_data
    os.remove(f"{test_filename}.csv")  # clean up the test file


def test_logging():
    with patch("logging.Logger.info") as mock_info, patch.object(
        AnkiCardGenerator, "_get_chain"
    ) as mock_get_chain:
        # Create a mock chain that returns a predictable result
        mock_chain = MagicMock()
        mock_chain.return_value = {"text": "test_result"}
        mock_get_chain.return_value = mock_chain

        generator = AnkiCardGenerator()
        result = generator._generate("test_subject", "details")
        assert mock_info.call_count == 3  # check if info was called 3 times
        assert result == "test_result"


def test_log_result():
    with patch("logging.Logger.info") as mock_info:
        generator = AnkiCardGenerator()
        result = "Line 1\nLine 2\nLine 3"
        generator.log_result(result, verbose=True)
        assert mock_info.call_count == 3  # check if info was called 3 times
