import unittest
from unittest.mock import mock_open, patch

import convert_csv_to_json


class TestGetFileData(unittest.TestCase):
    def test_get_anki_file_data(self):
        mocked_file_content = "DeckName\nFront1,Back1\nFront2,Back2"
        m = mock_open(read_data=mocked_file_content)

        with patch("convert_csv_to_json.open", m):
            result = convert_csv_to_json.get_file_data("mocked_path", "anki_v1.csv")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["deckName"], "DeckName")
        self.assertEqual(len(result[0]["cards"]), 2)

    def test_get_prose_file_data(self):
        mocked_file_content = "This is a prose text."
        m = mock_open(read_data=mocked_file_content)

        with patch("convert_csv_to_json.open", m):
            result = convert_csv_to_json.get_file_data("mocked_path", "prose_v1.txt")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["chapterName"], "mocked_path")
        self.assertEqual(result[0]["text"], "This is a prose text.")


class TestProcessDirectory(unittest.TestCase):
    @patch(
        "convert_csv_to_json.os.listdir",
        return_value=["file1_anki_v1.csv", "file2_anki_v1.csv"],
    )
    @patch(
        "convert_csv_to_json.get_file_data",
        return_value=[
            {"deckName": "DeckName", "cards": [{"front": "Front1", "back": "Back1"}]}
        ],
    )
    @patch("convert_csv_to_json.open", new_callable=mock_open)
    def test_process_directory_anki(
        self, mock_open_instance, mock_get_file_data, mock_listdir
    ):
        convert_csv_to_json.process_directory(
            "mocked_directory", "mocked_output.json", "anki_v1.csv"
        )

        mock_open_instance.assert_called_once_with("mocked_output.json", "w")

    @patch(
        "convert_csv_to_json.os.listdir",
        return_value=["file1_prose_v1.txt", "file2_prose_v1.txt"],
    )
    @patch(
        "convert_csv_to_json.get_file_data",
        return_value=[{"chapterName": "ChapterName", "text": "This is a prose text."}],
    )
    @patch("convert_csv_to_json.open", new_callable=mock_open)
    def test_process_directory_prose(
        self, mock_open_instance, mock_get_file_data, mock_listdir
    ):
        convert_csv_to_json.process_directory(
            "mocked_directory", "mocked_output.json", "prose_v1.txt"
        )

        mock_open_instance.assert_called_once_with("mocked_output.json", "w")


if __name__ == "__main__":
    unittest.main()
