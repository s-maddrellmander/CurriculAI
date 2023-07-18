import io
import os
from unittest import mock
from unittest.mock import Mock, mock_open, patch

import pytest

import anki
from large_content_generator import LargeContentGenerator


def test_read_subjects(tmp_path):
    gen = LargeContentGenerator()

    # Create a temporary file and write some content to it
    file_path = tmp_path / "subjects.txt"
    file_path.write_text("topic1\ntopic2\n")

    subjects = gen.read_subjects(file_path)
    assert subjects == ["topic1", "topic2"]


@patch("large_content_generator.AnkiCardGenerator")
def test_generate_and_save(mocked_anki):
    gen = LargeContentGenerator()
    gen.save_to_file = Mock()  # Mock the save_to_file function

    # Mock the generate and generate_MCQs functions of AnkiCardGenerator
    mocked_anki.return_value.generate.return_value = {}
    mocked_anki.return_value.generate_MCQs.return_value = ({}, "")

    with patch("large_content_generator.tqdm") as mocked_tqdm:
        mocked_tqdm.side_effect = lambda x, **kwargs: x  # Bypass tqdm
        gen.generate_and_save("dummy_subject")

    assert (
        gen.save_to_file.call_count == 9
    )  # Check if function was called 15 times (5 for each content type)


@patch("large_content_generator.AnkiCardGenerator")
def test_save_to_file(mocked_anki):
    gen = LargeContentGenerator()
    data = {"dummy": "data"}
    subject = "dummy subject"
    file_prefix = "dummy_file"

    with patch("builtins.open", mock_open()) as mock_file:
        gen.save_to_file(data, subject, file_prefix)

    filename = f"{subject.replace(' ', '_')}_{file_prefix}.json"
    subject_dir = os.path.join(gen.data_path, subject.replace(" ", "_"))
    mock_file.assert_called_once_with(os.path.join(subject_dir, filename), "w")


def test_print_summary():
    gen = LargeContentGenerator()
    gen.summary_data = {
        "dummy_subject": {"prose": 5, "anki": 5, "mcq": 5, "time": 0.01}
    }
    summary_str = gen.print_summary(testing=True)
    assert "Subject: dummy_subject" in summary_str
    assert "Generated 5 prose documents" in summary_str
    assert "Generated 5 anki card sets" in summary_str
    assert "Generated 5 mcq sets" in summary_str
    assert "Time taken: 0.01 seconds" in summary_str
