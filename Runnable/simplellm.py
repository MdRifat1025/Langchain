from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


load_dotenv()

model=ChatGroq(
   model="llama-3.1-8b-instant",temperature=0.7
)

prompt=PromptTemplate(
    template="suggest a cathcy blog title {topic}",
    input_variables=['topic']
)

topic=input("Enter the topic of the blog: ")

formated_prompt=prompt.format(topic=topic)

blog_title=model.predict(formated_prompt)

print(blog_title)   