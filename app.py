#import the env variables
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

#load pdf
from langchain_community.document_loaders import PyPDFLoader
path = r"D:\PROJECTS\RAG-chatbot\data\uploads\adapting_to_kannada_agent.pdf"
loader = PyPDFLoader(path)
docs = loader.load()


#text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)


#vector embeddings


