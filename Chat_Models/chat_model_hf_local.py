from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv

llm = HuggingFacePipeline.from_model_id(
    model_id= "HuggingFaceTB/SmolLM2-360M-Instruct",
    task = "text-generation",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 100
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Tell Me Places to visist in Bangalore")

print(result.content)