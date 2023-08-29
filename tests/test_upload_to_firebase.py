from unittest.mock import Mock, mock_open, patch

import pytest

from upload_to_firebase import (  # Adjust the import accordingly
    add_card_to_db,
    add_deck_to_db,
    load_json_data,
)


def test_load_json_data():
    with patch(
        "builtins.open", mock_open(read_data='{"deck": [{"deckName": "Test"}]}')
    ):
        data = load_json_data("dummy_path.json")
        assert data == {"deck": [{"deckName": "Test"}]}


def test_add_deck_to_db():
    mock_db = Mock()
    mock_db.collection().add.return_value = [None, Mock(id="12345")]

    deck_ref = add_deck_to_db(mock_db, "Test Deck")
    assert deck_ref.id == "12345"


def test_add_card_to_db():
    mock_db = Mock()
    mock_deck_ref = Mock(id="12345")

    card_data = {"front": "Question?", "back": "Answer."}

    add_card_to_db(mock_db, mock_deck_ref, card_data)

    mock_db.collection().add.assert_called_with(
        {"deckId": "12345", "front": "Question?", "back": "Answer.", "score": 2}
    )
