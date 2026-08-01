from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatOpenAI()

class FeedbackResponse(BaseModel):

    sentiment: Literal["positive","negative"] = Field("Classify the sentiment of the feedback.")

response_feedback = PydanticOutputParser(pydantic_object=FeedbackResponse)

general_parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Classify the Sentiment of the feedback. \n {feedback} \n {format_instruction}.",
    input_variables=["feedback"],
    partial_variables={"format_instruction" : response_feedback.get_format_instructions()}
)

classifier_chain = prompt1 | model | response_feedback

positive_prompt = PromptTemplate(
    template="Generate an appropriate respoonse for this feeback. \n {feedback}",
    input_variables=["feedback"]
)

negative_prompt = PromptTemplate(
    template="Generate an appropriate response for this feeback. \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive" , positive_prompt | model | general_parser),
    (lambda x : x.sentiment == "negative" , negative_prompt | model | general_parser),
    # default chain
    RunnableLambda(lambda x : "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback" : "This is the worst SmartPhone."})

print(result)

chain.get_graph().print_ascii()


