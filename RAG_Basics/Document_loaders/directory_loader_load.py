from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent.parent

loader = DirectoryLoader(
    path=BASE_DIR / 'directory_loader_understanding',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))
