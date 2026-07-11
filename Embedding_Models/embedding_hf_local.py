from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name = model_name)

text = "Delhi is the Capital of India!"

result = embedding.embed_query(text)

print(result)