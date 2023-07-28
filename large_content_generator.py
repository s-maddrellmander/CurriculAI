import json
import os
import time
from typing import List

from tqdm import tqdm

from anki import AnkiCardGenerator


class LargeContentGenerator:
    def __init__(self, model_name="gpt-3.5-turbo-16k-0613"):
        self.generator = AnkiCardGenerator(model_name=model_name)
        self.data_path = "data"
        self.notes = "Generate advanced level content, intended for postgraduate expert students."
        self.summary_data = {}  # Add a summary_data attribute

    def read_subjects(self, filename):
        with open(filename, "r") as f:
            return [line.strip() for line in f.readlines()]

    def generate_and_save(self, subject):
        self.summary_data[subject] = {
            "prose": 0,
            "anki": 0,
            "mcq": 0,
        }
        start_time = time.time()

        # Generate and save 5 versions of each type of content
        for i in tqdm(range(1, 4), desc="Temperature Iteration:"):
            # Reset the history
            self.generator.questions_asked = set()
            # Prose
            prose = self.generator.generate(subject, self.notes, format="prose")
            self.save_to_file(prose, subject, f"prose_v{i}")
            self.summary_data[subject]["prose"] += 1

            # Anki cards
            anki = self.generator.generate(
                subject, self.notes, format="anki", extra="", verbose=False
            )
            self.save_to_file(anki, subject, f"anki_v{i}")
            self.summary_data[subject]["anki"] += 1
            # MCQs
            mcq, _ = self.generator.generate_MCQs(subject, self.notes)
            self.save_to_file(mcq, subject, f"mcq_v{i}")
            self.summary_data[subject]["mcq"] += 1

        self.summary_data[subject]["time"] = round(time.time() - start_time, 2)

    def save_to_file(self, data, subject, file_prefix):
        filename = f"{subject.replace(' ', '_')}_{file_prefix}.json"
        subject_dir = os.path.join(self.data_path, subject.replace(" ", "_"))

        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)

        with open(os.path.join(subject_dir, filename), "w") as outfile:
            json.dump(data, outfile)

    def print_summary(self, testing=False):  # Add a testing parameter
        summary_str = "\nSummary of generated content:"
        for subject, data in self.summary_data.items():
            summary_str += f"\n\nSubject: {subject}"
            summary_str += f"\nGenerated {data['prose']} prose documents"
            summary_str += f"\nGenerated {data['anki']} anki card sets"
            summary_str += f"\nGenerated {data['mcq']} mcq sets"
            summary_str += f"\nTime taken: {data['time']} seconds"
        if testing:
            return summary_str
        else:
            print(summary_str)

    def improve_content(self, subject: str, file_versions: List[str]):
        content_by_type = {"prose": [], "anki": [], "mcq": []}

        # Load the content from the specified file versions and sort by type
        for content_type in content_by_type.keys():
            for version in file_versions:
                filename = f"{subject.replace(' ', '_')}_{content_type}_{version}.json"
                subject_dir = os.path.join(self.data_path, subject.replace(" ", "_"))

                with open(os.path.join(subject_dir, filename), "r") as file:
                    content = json.load(file)
                    content_by_type[content_type].append(content)

        # Use the generator's combine function to generate improved content for each type
        for content_type, contents in content_by_type.items():
            if contents:  # Check if contents exist
                # Combine the contents
                improved_content = self.generator.combine(contents)

                # Save the improved content to a new file
                file_prefix = f"{content_type}_improved"
                self.save_to_file(improved_content, subject, file_prefix)

        print(f"Improved content for {subject} has been generated and saved.")


if __name__ == "__main__":
    generator = LargeContentGenerator()
    subjects = generator.read_subjects("short_subjects.txt")
    for subject in tqdm(subjects, desc="Sylabus Progress:"):
        generator.generate_and_save(subject)
        generator.improve_content(subject, ["v1", "v2", "v3"])
    generator.print_summary()
