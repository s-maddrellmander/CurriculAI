import json

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("quiz-app-f9560-firebase-adminsdk-vxdaq-469cd9ab06.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Read your structured data (e.g., from a JSON file)
with open("your_data.json", "r") as f:
    data = json.load(f)

for deck in data:
    deck_ref = db.collection("decks").add(
        {"name": deck["deckName"], "description": "Sample description"}
    )[
        1
    ]  # Get the reference to the newly created deck

    for card in deck["cards"]:
        db.collection("cards").add(
            {
                "deckId": deck_ref.id,
                "front": card["front"],
                "back": card["back"],
                "score": 2,  # Default score
            }
        )
