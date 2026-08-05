from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a koke on {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain the {joke}",
    input_variables=["joke"]
)

joke_gen_chain = RunnableSequence(prompt1,model,parser)

parallell_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2,model,parser)
})

chain = RunnableSequence(joke_gen_chain,parallell_chain)

result = chain.invoke({"topic":"AI"})

print(result)