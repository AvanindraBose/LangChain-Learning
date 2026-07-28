from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

class Review(TypedDict):
    summary: str
    sentiment: str

struct_model = model.with_structured_output(Review)

result = struct_model.invoke('''The hardware is great, but the software feels bloated.
There are too many pre-installed apps that I can't remove. Also, the UI looks outdated
compared to other brands. Hoping for a software update to fix this.''')

print(result)
print(result['summary'])
print(result['sentiment'])