import json

import firebase_admin
from firebase_admin import credentials, firestore


def initialize_firebase(cred_path):
    """
    Initialize Firebase with given credentials.
    """
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    return firestore.client()


def load_json_data(file_path):
    """
    Load data from a JSON file.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def add_deck_to_db(db, deck_name, description="Sample description"):
    """
    Add a deck to the database.
    """
    deck_ref = db.collection("decks").add(
        {"name": deck_name, "description": description}
    )[1]
    return deck_ref


def add_card_to_db(db, deck_ref, card, score=2):
    """
    Add a card to the database.
    """
    db.collection("cards").add(
        {
            "deckId": deck_ref.id,
            "front": card["front"],
            "back": card["back"],
            "score": score,
        }
    )


def main(
    cred_path="quiz-app-f9560-firebase-adminsdk-vxdaq-469cd9ab06.json",
    data_path="your_data.json",
):
    db = initialize_firebase(cred_path)
    data = load_json_data(data_path)

    for deck in data:
        deck_ref = add_deck_to_db(db, deck["deckName"])
        for card in deck["cards"]:
            add_card_to_db(db, deck_ref, card)


if __name__ == "__main__":
    main()
