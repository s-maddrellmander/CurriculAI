# CurriculAI

[![codecov](https://codecov.io/gh/s-maddrellmander/CurriculAI/branch/main/graph/badge.svg?token=XU26BNTC8I)](https://codecov.io/gh/s-maddrellmander/CurriculAI)
[![test](https://github.com/s-maddrellmander/CurriculAI/actions/workflows/python-app.yml/badge.svg)](https://github.com/s-maddrellmander/CurriculAI/actions/workflows/python-app.yml)
[![black lint](https://github.com/s-maddrellmander/CurriculAI/actions/workflows/lint.yml/badge.svg)](https://github.com/s-maddrellmander/CurriculAI/actions/workflows/lint.yml)


CurriculAI is an AI-based educational tool that leverages the power of Langchain and OpenAI LLMs to transform user resources into accessible and engaging learning materials.

## Quick Start
- **Set up environment**
```bash
git clone https://github.com/s-maddrellmander/CurriculAI.git
cd CurriculAI
python3 -m virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

- **Generate Sylabus**
    - Generate a `sylabus.txt` file. 
    - Assume the list will look like the example below:

    (`short_subjects.txt`)
    ```
    1. Introduction to Machine Learning
    2. Basics of Python for Machine Learning
    3. Supervised Learning
    4. Unsupervised Learning
    ```

- **Generate Content**
    ```bash
    python large_content_gen.py
    ```
    - This will generate by default prose and flashcard style questions on each topic / heading in the sylabus file

- **(Optional) Convert format**
    ```bash
    python convert_csv_to_json.py
    ```
    - This converts all the saved csv files into appropriate JSON files for flashcard use later
    - `WARNING` It will iteratively step through all directories in the saved data location and pull all csv files.
- **(Optional) Sync with database**
    ```bash
    python upload_to_firebase.py
    ```
    - This assumes you have the necessary firebase authetication file locally in the directory - this needs to be updated.

## Features

- **Vectorised Database**: CurriculAI tokenises and stores all relevant resources into a vectorised database, ensuring efficient information retrieval and content generation.

- **Customisable Prompts**: The tool can generate textbook-style content and shorter form notes based on the custom prompts provided by the user. 

- **Question and Flash Card Generation**: Besides content generation, CurriculAI also has the capability to generate multiple-choice questions and flashcards, offering a comprehensive learning experience.

## How it works

### Vectorised Database
CurriculAI starts by tokenising user resources and storing them into a vectorised database. This process allows for efficient access and manipulation of information, facilitating subsequent content generation.

### Customisable Prompts
Once the resources are in the database, users can utilise custom prompts to generate textbook-style content or shorter notes. This feature enhances the tool's flexibility and usability, catering to a wide range of educational needs.

### Question and Flash Card Generation
In addition to content generation, CurriculAI can generate multiple-choice questions and flashcards from the same resources. This allows for a more interactive and engaging learning process.

## Technologies Used

- **Langchain**: Langchain is used for tokenising resources and processing language data.
- **OpenAI LLMs**: The content and question generation aspects are powered by OpenAI's Language Model (LLM).

## Future Development
We have ambitious plans for CurriculAI, including enhancing the AI's understanding capabilities, broadening the range of content that can be generated, and refining the user interface for an even smoother user experience.

Join us in revolutionizing the way we create educational resources!

