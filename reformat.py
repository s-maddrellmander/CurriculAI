"""
Simple reformatting file
"""

import json
import csv
from typing import Any, Dict


class JSONFormatter:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_json(self) -> Dict[str, Any]:
        with open(self.filepath, 'r') as file:
            data = json.load(file)
        return data

    def save_as_txt(self, data: str, filename: str) -> None:
        with open(filename, 'w') as file:
            file.write(data)

    def save_flashcards_as_csv(self, flashcards: Dict[str, Any], filename: str) -> None:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Question", "Answer"])  # header
            for card in flashcards:
                writer.writerow([card["question"], card["answer"]])

    def save_mcq_as_csv(self, mcqs: Dict[str, Any], filename: str) -> None:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Question", "Options", "Answer"])  # header
            for question in mcqs:
                writer.writerow([question["question"], ", ".join(question["options"]), question["answer"]])

    def process_data(self, text_filename: str, flashcard_filename: str, mcq_filename: str) -> None:
        data = self.load_json()

        if 'prose' in data:
            self.save_as_txt(data['prose'], text_filename)
        if 'flashcards' in data:
            self.save_flashcards_as_csv(data['flashcards'], flashcard_filename)
        if 'mcq' in data:
            self.save_mcq_as_csv(data['mcq'], mcq_filename)



