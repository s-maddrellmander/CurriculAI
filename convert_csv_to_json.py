import csv
import json
import os

data_directory = "data"
output_data = []

# Iterate through each folder in the data directory
for folder_name in os.listdir(data_directory):
    folder_path = os.path.join(data_directory, folder_name)

    # Check if the path is a directory
    if os.path.isdir(folder_path):
        # Extract deck name from folder name
        deck_name = folder_name.split("_", 1)[-1].replace("_", " ")

        # Iterate through each file in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith("anki_v1.csv"):
                file_path = os.path.join(folder_path, file_name)

                # Read CSV file and extract cards
                with open(file_path, "r", encoding="utf-8") as csv_file:
                    reader = csv.reader(csv_file, delimiter=";")
                    cards = [{"front": row[0], "back": row[1]} for row in reader]

                # Add deck and cards to the output data
                output_data.append({"deckName": deck_name, "cards": cards})

# Write JSON file
with open("your_data.json", "w", encoding="utf-8") as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)
