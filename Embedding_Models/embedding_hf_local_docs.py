from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name = model_name)

doc = [
    "India is the Best Country",
    "Virat Kolhi is the Best Cricketer",
    "Good Things Takes Time!"
]

result = embedding.embed_documents(doc)

print(result)