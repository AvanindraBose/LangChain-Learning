from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = CharacterTextSplitter(
    chunk_size = 2000,
    chunk_overlap = 0,
    separator=""
)

loader = PyPDFLoader(file_path='dl-curriculum.pdf')

docs = loader.lazy_load()

text = splitter.split_documents(docs)

print(text[0].metadata)