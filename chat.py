import os

os.environ["OPENAI_API_KEY"] = "YOUR API KEY"

from langchain.document_loaders import PyPDFDirectoryLoader


def chat():
    pdf_folder_path = "~/Zotero/storage/*"
    loader = PyPDFDirectoryLoader(pdf_folder_path)
    docs = loader.load()
    print(len(docs))
