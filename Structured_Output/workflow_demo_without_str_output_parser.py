from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="Text Generation",
    temperature=0.2
)

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
    template="Write in detail about the {topic}.",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary for the given {description}.",
    input_variables=["description"]
)

prompt1 = template1.invoke({"topic":"HarmanPreet Kaur"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"description":result.content})

final_result = model.invoke(prompt2)

print(final_result.content)

