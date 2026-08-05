from langchain_core.runnables import RunnableSequence,RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "write a tweet in 50 chars about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="write a reddit post in 50 chars about {topic}",
    input_variables=["topic"]
)

chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1,model,parser),
    "reddit" : RunnableSequence(prompt2,model,parser)
})

result = chain.invoke("AI")

print(result)