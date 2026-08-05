from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent.parent

loader = DirectoryLoader(
    path=BASE_DIR / 'directory_loader_understanding',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

print(type(docs))

for _ in range(2):
    print(next(docs).page_content)