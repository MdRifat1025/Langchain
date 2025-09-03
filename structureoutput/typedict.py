from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

class Review(TypedDict):
    summary: str
    sentiment: str

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
Gowtam Tinnanuri, known for his
         emotionally engaging storytelling in Jersey, brings a similar sensibility to Kingdom. 
    Though mounted on a broader canvas with action and fictional history woven in, the film never loses sight of its emotional core. Kingdom taps into a classic saviour myth, where a displaced people hold on to ancestral belief that one day, someone bearing the signs of their ancient wisdom will lead them home.
""")

print(result)
