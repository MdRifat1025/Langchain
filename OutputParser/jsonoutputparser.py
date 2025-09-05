from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

from langchain.prompts import PromptTemplate

load_dotenv()

parser = JsonOutputParser()


# Format instructions
format_instructions = parser.get_format_instructions()

# Model
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

prompt=PromptTemplate(
    template="""extract  5 sentece from cricket.{format_instructions}""",
    input_variables=["text"],
    partial_variables={"format_instructions":format_instructions}
)

# Chain তৈরি
chain = prompt | model | parser

# Run
result = chain.invoke({})

print(result)
