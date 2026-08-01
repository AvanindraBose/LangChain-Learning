from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7
)

model2 = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate Notes in 100 words for the following document /n {document}.",
    input_variables=["document"]
)

prompt2 = PromptTemplate(
    template = "Generate 5 questions to test the understanding of the document. /n {document}",
    input_variables=["document"]
)

prompt3 = PromptTemplate(
    template="Merge the outputs in a single document. notes -> {notes} quiz -> {quiz}.",
    input_variables=["notes","quiz"]
)

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz" : prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

result = chain.invoke({"document": "black hole"})

print(result)

chain.get_graph().print_ascii()

