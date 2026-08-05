from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='csv_loader_understanding.csv')

docs = loader.load()

#  Every Row will be converted into a document object

print(len(docs))