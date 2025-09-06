from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model=ChatGroq(
    model="llama-3.1-8b-instant",temperature=0.7
)

prompt=PromptTemplate(
    template="What is a good name for a company that makes {product}?",
    input_variables=["product"])
parser=StrOutputParser()
    

chain=prompt |model| parser
result=chain.invoke({"product":"colorful_sockes"})

print(result)

