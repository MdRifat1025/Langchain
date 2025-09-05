from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
load_dotenv()

parser=StrOutputParser()

model=ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

prompt=PromptTemplate(
     template="""
extract student name and age from the following text:{text}
""",
input_variables=["text"]
)

chain=prompt |model |parser

result=chain.invoke({"text":"student name is Rifat and age is 23"})


print(result)