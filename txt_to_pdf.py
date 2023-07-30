import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def custom_sort(s):
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


# Read subjects.txt file and prepare the list of directories
with open("subjects_enzymes.txt", "r") as f:
    directories = sorted(
        [line.strip().replace(" ", "_") for line in f], key=custom_sort
    )

# Set the parent directory path
parent_directory = "data/"

doc = SimpleDocTemplate("output.pdf", pagesize=letter)
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
            story.append(Paragraph(text, styles["BodyText"]))  # Add the file contents
            story.append(Spacer(1, 12))  # Add a spacer

doc.build(story)

print("PDF created successfully.")
