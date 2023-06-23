import os
import pytest
from unittest.mock import MagicMock
import os
import pytest
from documents import vector_embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from unittest.mock import patch

from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader

from documents import get_docs_for_QA, load_pdf_pages
import pytest

from documents import get_docs_for_question_gen, load_pdf_pages


@pytest.mark.parametrize(
    "file_path,expected",
    [
        (
            "tests/resources/the-odyssey.pdf",
            "Download free eBook s of classic literature, books and",
        ),
        # add as many test cases as you like
    ],
)
def test_pdf_to_text(file_path: str, expected: str):
    assert expected in load_pdf_pages(file_path)[:100]


@pytest.mark.parametrize(
    "text, expected_length",
    [
        ("Test Text " * 1100, 3),
        ("Test Text " * 200, 1),
        ("Test Text " * 5500, 13),
        ("", 0),  # empty text
    ],
)
def test_get_docs_for_question_gen(text, expected_length):
    docs = get_docs_for_question_gen(text, chunk_size=1000, chunk_overlap=100)
    assert len(docs) == expected_length


@patch("documents.logger")  # replace with the actual module
def test_get_docs_for_question_gen_logs(mock_logger):
    text = "This is a test text. " * 1000
    get_docs_for_question_gen(text)
    assert mock_logger.info.call_count == 2
    mock_logger.info.assert_any_call("Splitting text for question gen")
    mock_logger.info.assert_any_call("Generating Documents")


class MockPyPDFLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        return ["This is a test page.", "This is another test page."]


def test_get_docs_for_QA(monkeypatch):
    # Arrange
    expected_result = ["This is a test page.", "This is another test page."]
    test_file_path = "test_path"

    # apply the monkeypatch for PyPDFLoader
    monkeypatch.setattr("documents.PyPDFLoader", MockPyPDFLoader)

    # Act
    result = get_docs_for_QA(test_file_path)

    # Assert
    assert result == expected_result, f"Expected {expected_result}, but got {result}"


class MockPage:
    def extract_text(self):
        return "This is a test page."


class MockPdfReader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.pages = [MockPage(), MockPage()]


def test_load_pdf_pages(monkeypatch):
    # Arrange
    expected_result = "This is a test page.This is a test page."
    test_file_path = "test_path"

    # apply the monkeypatch for PdfReader
    monkeypatch.setattr("documents.PdfReader", MockPdfReader)

    # Act
    result = load_pdf_pages(test_file_path)

    # Assert
    assert result == expected_result, f"Expected {expected_result}, but got {result}"
