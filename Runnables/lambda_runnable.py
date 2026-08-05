# Runnable Primitives
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a joke on {topic}",
    input_variables=["topic"]
)

def word_counter(text):
    return len(text.split())

joke_gen_cahin = RunnableSequence(prompt1,model,parser)

p_c = RunnableParallel({
    'joke': RunnablePassthrough(),
    'length' : RunnableLambda(word_counter)
})

chain = RunnableSequence(joke_gen_cahin,p_c)

result = chain.invoke({"topic":"Human Values"})

print(result)