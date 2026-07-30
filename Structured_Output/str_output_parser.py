from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="Text Generation",
    temperature=0.2
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write in detail about the {topic}.",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary for the given {description}.",
    input_variables=["description"]
)

parser = StrOutputParser()

# Using Chains: this is a future concept

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic" : "Virat Kohli"})

print(result)
