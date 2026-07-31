from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name : str = Field(description="Name of the Person")
    age : int = Field(gt=18,description="Age of the person strictly following the condition")
    city: str = Field(description="City from which the person belongs")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template=''' Generate the name, age , city of the fictional {place} person

    {format_instruction}
''',
input_variables=["place"],
partial_variables={"format_instruction":parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({"place":"American"})

print(final_result)
