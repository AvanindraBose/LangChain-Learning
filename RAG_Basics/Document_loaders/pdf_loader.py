from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

loader = PyPDFLoader(BASE_DIR / 'pdf_loader_understanding.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
