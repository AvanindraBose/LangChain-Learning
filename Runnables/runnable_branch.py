from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableBranch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "generate a report on {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a summary on {report}",
    input_variables=["report"]
)

report_gen_chain = RunnableSequence(prompt1 , model , parser)

branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 500 , RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain,branch_chain)

result = final_chain.invoke({"topic":"Indina GDP"})

print(result)