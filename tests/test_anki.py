import logging
import os
from unittest.mock import patch

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
    with patch("logging.Logger.info") as mock_info:
        generator = AnkiCardGenerator()
        result = generator._generate("test_subject", "details")
        assert mock_info.call_count == 3  # check if info was called 4 times
        # TODO: Add test for the output format here as well.


def test_log_result():
    with patch("logging.Logger.info") as mock_info:
        generator = AnkiCardGenerator()
        result = "Line 1\nLine 2\nLine 3"
        generator.log_result(result, verbose=True)
        assert mock_info.call_count == 3  # check if info was called 3 times
