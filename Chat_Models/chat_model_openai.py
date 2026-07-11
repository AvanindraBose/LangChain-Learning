from langchain_openai import ChatOpenAI
from dotenv  import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4',temperature=1.7)

result = model.invoke("Write a 5 line poem on Rohit Sharma.")

print(result.content)