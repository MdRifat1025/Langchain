from pydantic import BaseModel, Field
from typing import Literal,Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


model=ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either negative, positive or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    

structured_model=model.with_structured_output(Review)

result=structured_model.invoke("""
Extract the name of the student from the following text. Return only valid JSON. """)

print(result)