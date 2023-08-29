import csv
import json
import os


def get_file_data(file_path, file_type):
    """
    Reads the file content and returns data in desired format.
    """
    data = []

    if file_type == "anki_v1.csv":
        with open(file_path, "r") as f:
            reader = csv.reader(f)
            deck_name = next(reader)[0]
            cards = [{"front": row[0], "back": row[1]} for row in reader]
            data.append({"deckName": deck_name, "cards": cards})

    elif file_type == "prose_v1.txt":
        with open(file_path, "r") as f:
            text = f.read()
            chapter_name = os.path.basename(file_path).replace("_prose_v1.txt", "")
            data.append({"chapterName": chapter_name, "text": text})

    return data


def process_directory(folder_path, output_file, file_suffix):
    """
    Iterates through the directory, reads files with desired suffix, and
    aggregates the data to write to an output file.
    """
    all_data = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(file_suffix):
            file_path = os.path.join(folder_path, file_name)
            all_data.extend(get_file_data(file_path, file_suffix))

    with open(output_file, "w") as out:
        json.dump(all_data, out)


def main(folder_path):
    process_directory(folder_path, "anki_data.json", "anki_v1.csv")
    process_directory(folder_path, "prose_data.json", "prose_v1.txt")


if __name__ == "__main__":
    import sys

    try:
        folder_path = sys.argv[1]
    except IndexError:
        print("Usage: python script_name.py <FOLDER_PATH>")
        sys.exit(1)

    main(folder_path)
