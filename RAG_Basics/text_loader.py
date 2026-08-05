from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

loader = TextLoader('text_loader_understanding.txt',encoding='utf-8')

docs = loader.load()

print(type(docs[0]))
print(docs[0].metadata)
print(docs[0].page_content)

model = ChatOpenAI()
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Summarize this document. \n {document}",
    input_variables=["document"]
)

chain = prompt | model | parser

result = chain.invoke({"document": docs[0].page_content})

print(result)


