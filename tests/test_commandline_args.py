import pytest
from unittest.mock import patch
import main  # replace with the actual name of your script


def test_parse_arguments():
    with patch(
        "sys.argv",
        [
            "main",
            "--key",
            "test_key",
            "--question_answer",
            "--prose_generation",
            "--subject",
            "Test Subject",
        ],
    ):
        args = main.parse_arguments()
        assert args.key == "test_key"
        assert args.question_answer is True
        assert args.prose_generation is True
        assert args.subject == "Test Subject"
