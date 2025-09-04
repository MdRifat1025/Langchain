from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Literal,Optional

load_dotenv()

class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos", "neg"], "Return sentiment of the review either negative, positive or neutral"]
    pros: Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name: Annotated[Optional[str], "Write the name of the reviewer"]

model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

structured_model = model.with_structured_output(Review)

result =structured_model.invoke("""
Analyze the following review. Extract key themes, write a brief summary, determine sentiment 
(positive, negative, or neutral), list pros and cons, and include the reviewer name if available.
Return only valid JSON.

Review text:
Gowtam Tinnanuri, known for his emotionally engaging storytelling in Jersey, brings a similar sensibility to Kingdom. 
Though mounted on a broader canvas with action and fictional history woven in, the film never loses sight of its emotional core. 
Kingdom taps into a classic saviour myth, where a displaced people hold on to ancestral belief that one day, someone bearing the signs of their ancient wisdom will lead them home.
""")


print(result)
