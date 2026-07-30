from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="Text Generation",
    temperature=0.2
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template='''Give me the details of the author of this book {book_name}

    {format_instructions}''',
    input_variables=["book_name"],
    partial_variables={"format_instructions" : parser.get_format_instructions()}
)

prompt = template.invoke({"book_name":"Harry Potter"})

result = model.invoke(prompt)

final_result = parser.parse(result.content)
print(final_result)
print(type(final_result))