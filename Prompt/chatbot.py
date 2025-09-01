from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()


model=ChatGroq(model="llama-3.1-8b-instant",temperature=0.7)

history=[]
while True:
    user_input=input("Human:")
    history.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    result=model.invoke(history)
    history.append(AIMessage(content=result.content))
    print("AI:",result.content)
print(history)
