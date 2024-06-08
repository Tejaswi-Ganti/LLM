import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings  
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import PyPDFDirectoryLoader 
import time 

from dotenv import load_dotenv
load_dotenv()

#load thre GROQ API key

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')
##os.environ['GROQ_API_KEY']

st.title('ChatGroq with Llama3')


llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

# if "vector" not in st.session_state:
#     st.session_state.embeddings=OllamaEmbeddings()
#     st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")   ###Data Ingesion
#     st.session_state.docs=st.session_state.loader.load() ##Data Loading 

#     st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) #Chunk
#     st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
#     st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

## st.title("ChatGroq Demo")

##llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma-7b-It")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

## prompt1=st.text_input("Input you prompt here")

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading 

        # Check if documents are loaded
        if not st.session_state.docs:
            st.error("No documents loaded. Check the file path and content.")
            return

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting

        # Check if documents are split
        if not st.session_state.final_documents:
            st.error("Document splitting failed. Check the document content and splitting logic.")
            return

        # Generate embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI Embeddings

prompt1=st.text_input("Input you prompt here")

if st.button("Documents Embeddings"):
    vector_embedding()
    st.write("Vector Store DB is Ready")

import time


if prompt1:
    start=time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response=retrieval_chain.invoke({"input":prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
    
# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS 
# from langchain_community.document_loaders import PyPDFDirectoryLoader 
# import time 

# from dotenv import load_dotenv
# load_dotenv()

# # Load the Groq API key
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# groq_api_key = os.getenv('GROQ_API_KEY')

# st.title('ChatGroq with Llama3')

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# prompt = ChatPromptTemplate.from_template("""
# Answer the questions based on the provided context only.
# Please provide the most accurate response based on the question
# <context>
# {context}
# <context>
# Questions:{input}
# """)

# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = OllamaEmbeddings()
#         st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
#         st.session_state.docs = st.session_state.loader.load()  # Document Loading 

#         # Check if documents are loaded
#         if not st.session_state.docs:
#             st.error("No documents loaded. Check the file path and content.")
#             return

#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting

#         # Check if documents are split
#         if not st.session_state.final_documents:
#             st.error("Document splitting failed. Check the document content and splitting logic.")
#             return

#         # Generate embeddings
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector OpenAI Embeddings

# prompt1 = st.text_input("Input your prompt here")

# if st.button("Documents Embeddings"):
#     vector_embedding()
#     st.write("Vector Store DB is Ready")

# if prompt1:
#     start = time.process_time()
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = st.session_state.vectors.as_retriever()
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
#     response = retrieval_chain.invoke({"input": prompt1})
#     print("Response time:", time.process_time() - start)
#     st.write(response['answer'])

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("--------------------------------")


