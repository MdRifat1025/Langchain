from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from langchain.prompts import PromptTemplate


#LOAD API KEY
load_dotenv()

class Student(BaseModel):
    name:Optional[str]=Field(None, description="name of the student")
    age:Optional[int]=Field(None,description="age of the student")

parser=PydanticOutputParser(pydantic_object=Student)

format_instructions=parser.get_format_instructions()

model=ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

prompt = PromptTemplate(
    template="""
Extract the student's name and age from the following text:
{text}

{format_instructions}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions}
)


chain=prompt |model |parser

result=chain.invoke({"text":"student name is Rifat and age is 23"})
print(result)