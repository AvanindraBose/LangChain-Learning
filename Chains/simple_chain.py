from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

template = PromptTemplate(
    template="Generate 3 interesting facts about {topic}.",
    input_variables=["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({"topic":"football"})

print(result)


print(chain.get_graph().draw_ascii())