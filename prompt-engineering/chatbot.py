from langchain_openai import ChatOpenAI

model = ChatOpenAI()

while True:
    user_input = input("You: ")
    if user_input == "exit":
        break
    result = model.invoke(user_input)
    print("AI: ",result.content)