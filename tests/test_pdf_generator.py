from unittest.mock import MagicMock, mock_open, patch

import pytest

from pdf_generator import custom_sort, main


# Test custom_sort function
def test_custom_sort():
    assert custom_sort("2.Some_Chapter") == (2, 0)
    assert custom_sort("1.1_Sub_Chapter") == (1, 1)
    assert custom_sort("11") == (11,)


# For the main test
@patch(
    "builtins.open", new_callable=mock_open, read_data="1.1_Sub_Chapter\n2.Some_Chapter"
)
@patch("pdf_generator.os.path.exists", return_value=True)
@patch("pdf_generator.SimpleDocTemplate.build")
def test_main(mock_build, mock_exists, mock_open_func):
    read_file_content = "Some dummy text."
    with patch(
        "pdf_generator.open", mock_open(read_data=read_file_content), create=True
    ):
        main("mocked_input.txt", "mocked_output.pdf")
    mock_build.assert_called_once()
