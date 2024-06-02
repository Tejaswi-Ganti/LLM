
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv


import streamlit as st
import os
from dotenv import load_dotenv

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

##Prompt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", " You are helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)


#streamlit framework

st.title('Langchain Demo with OPENAI API')
input_text = st.text_input("Search the topic u want")

#olama LLM
llm = Ollama(modedl = "llama2")
output_parser = StrOutputParser()
chain=prompt|llm|output_parser