from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

#  defintion: A MessagesPlaceholder is a placeholder inside a ChatPromptTemplate that lets you inject a list of chat messages (conversation history) at runtime.

# construct a template
chat_template = ChatPromptTemplate([
    ('system','You are an AI customer chatbot assistant.'),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human',"{query}")
])

# load the chat history
chat_history = []
with open('prompt-engineering/chathistory.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# invoke the prompt with user query

prompt = chat_template.invoke({
    'chat_history': chat_history,
    'query': "What is the status of the refund ?"
})

print(prompt)