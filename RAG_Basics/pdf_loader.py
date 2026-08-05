from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('pdf_loader_understanding.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)