from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32
)

documents = [
    "God Is Great!",
    "Thank You God for this Wonderful Life!",
    "I am doing Good now."
]

result = model.embed_documents(documents)

print(result)