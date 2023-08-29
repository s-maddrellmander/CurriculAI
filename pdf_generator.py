import argparse
import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def custom_sort(s):
    try:
        parts = s.split(".")
        if len(parts) > 1:
            subparts = parts[1].split("_")
            if subparts[
                0
            ].isdigit():  # check if the string is digit before converting to int
                return (int(parts[0]), int(subparts[0]))
            else:
                return (int(parts[0]), 0)
        else:
            return (int(parts[0]),)
    except ValueError:
        return (0, 0)  # or any other default value


def main(input_file, output_file):
    # Read input_file and prepare the list of directories
    with open(input_file, "r") as f:
        directories = sorted(
            [line.strip().replace(" ", "_") for line in f], key=custom_sort
        )

    # Set the parent directory path
    parent_directory = "data/"

    doc = SimpleDocTemplate(output_file, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    for dir_name in directories:
        # Construct the full path to the file
        file_path = os.path.join(parent_directory, dir_name, dir_name + "_prose_v1.txt")

        # Check if the file exists
        if os.path.exists(file_path):
            # Read the file and append its contents to the PDF
            with open(file_path, "r") as file:
                text = file.read().replace("\n", "<br/>")
                story.append(
                    Paragraph(dir_name.replace("_", " "), styles["Heading2"])
                )  # Add a heading
                story.append(
                    Paragraph(text, styles["BodyText"])
                )  # Add the file contents
                story.append(Spacer(1, 12))  # Add a spacer

    doc.build(story)

    print(f"PDF created successfully as {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a PDF from a list of directories."
    )
    parser.add_argument("input_file", help="Path to the input subjects file.")
    parser.add_argument(
        "-o", "--output", default="output.pdf", help="Path to the output PDF file."
    )

    args = parser.parse_args()
    main(args.input_file, args.output)
