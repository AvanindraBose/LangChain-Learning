from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report in 100 words for the {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Genarate a short summary for the {topic}",
    input_variables=["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Lionel Messi"})

print(result)

chain.get_graph().print_ascii()