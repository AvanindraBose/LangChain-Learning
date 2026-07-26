from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', "you are a {domain} expert"),
    ('human', "explain this {topic} in detail")
])

chat_template_new = ChatPromptTemplate.from_messages([
    ('system', "you are a {domain} expert"),
    ('human', "explain this {topic} in detail")
])

prompt = chat_template.invoke(
    {
        'domain':"Mathematics",
        'topic': 'Inverse Trignometry'
    }
)

print(prompt)