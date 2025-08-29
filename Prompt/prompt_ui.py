from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import load_prompt

load_dotenv()

model=ChatGroq(model="llama-3.1-8b-instant",temperature=0.7)
st.title("LangChain + Groq with Streamlit")

# Take user input
store=[]
user_input = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if user_input.strip():
       
        result = model.invoke(user_input)
    
        st.subheader("Response:")
        st.write(result.content)
    
    else:
        st.warning("Please enter a prompt before generating.")