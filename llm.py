from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

llm =OllamaLLM(model='llama3.2:3b')

result = llm.invoke("What is the capital of Bangladesh?")

print(result)