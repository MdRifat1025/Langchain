from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


model=ChatGroq(model="llama-3.1-8b-instant",temperature=0.7)

history=[]
while True:
    user_input=input("Human:")
    history.append(user_input)
    if user_input=="exit":
        break
    result=model.invoke(history)
    history.append(result.content)
    print("AI:",result.content)
print(history)
