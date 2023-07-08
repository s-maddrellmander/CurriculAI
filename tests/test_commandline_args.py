from unittest.mock import patch

import pytest

from main import parse_arguments


@pytest.mark.parametrize(
    "input_args,expected_output",
    [
        (
            [
                "main",
                "--key",
                "test_key",
                "--question_answer",
                "--prose_generation",
                "--subject",
                "Test Subject",
            ],
            {
                "key": "test_key",
                "question_answer": True,
                "prose_generation": True,
                "subject": "Test Subject",
                "chat": False,
                "save_questions": False,
            },
        ),
        (
            ["main", "--key", "test_key", "--chat", "--save_questions"],
            {
                "key": "test_key",
                "question_answer": False,
                "prose_generation": False,
                "subject": "Introduction to ML.",
                "chat": True,
                "save_questions": True,
            },
        ),
        (
            ["main", "--key", "test_key", "--subject", "Different Subject"],
            {
                "key": "test_key",
                "question_answer": False,
                "prose_generation": False,
                "subject": "Different Subject",
                "chat": False,
                "save_questions": False,
            },
        ),
        (
            ["main", "--key", "test_key"],
            {
                "key": "test_key",
                "question_answer": False,
                "prose_generation": False,
                "subject": "Introduction to ML.",
                "chat": False,
                "save_questions": False,
            },
        ),
    ],
)
def test_parse_arguments(input_args, expected_output):
    with patch("sys.argv", input_args):
        args = parse_arguments()
        assert args.key == expected_output["key"]
        assert args.question_answer == expected_output["question_answer"]
        assert args.prose_generation == expected_output["prose_generation"]
        assert args.subject == expected_output["subject"]
        assert args.chat == expected_output["chat"]
        assert args.save_questions == expected_output["save_questions"]
