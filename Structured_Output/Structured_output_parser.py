from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.2
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name = "Fact_1" , description="Fact1 about the topic"),
    ResponseSchema(name="Fact_2",description="Fact2 about the topic"),
    ResponseSchema(name="Fact_3",description="Fact3 about the topic")
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = """
    Give exactly three facts about {topic}.

    You MUST return the response using the provided JSON schema.

    Do not add any additional keys.
    Do not change the key names.

    {format_instructions} """,
    input_variables=["topic"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)

prompt = template.invoke({"topic":"black hole"})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)