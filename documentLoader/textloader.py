
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


model=ChatGroq(
   model="llama-3.1-8b-instant",temperature=0.7
)

prompt=PromptTemplate(
    template="What is the main topic of the following text? {text}",
    input_variables=["text"]
)
loader=TextLoader("documentLoader/DSA.txt", encoding="utf8")
parser=StrOutputParser()

chain=prompt|model|parser
docs=loader.load()
print(chain.invoke({'text':docs[0].page_content}))
