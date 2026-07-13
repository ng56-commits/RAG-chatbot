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


#vector embeddings and chroma db
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore.from_documents(
    embedding=embeddings,
    documents=docs
    )

#llm
from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-20b")

#tool
from langchain.tools import tool

@tool
def retrieve_context(query:str):
        """Retrieve documents relevant to a query from the knowledge base."""
        context = ""
        docs = vector_store.similarity_search(query=query, k=3)
        for doc in docs:
            context = doc.page_content + "\n\n"

        return context

#agent
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver 
system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
        My knowledge base consists of the details from the uploaded document. 
        ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""

from langgraph.checkpoint.memory import InMemorySaver  

agent_memory = InMemorySaver()

agent = create_agent(
      system_prompt=system_prompt,
      model=llm,
      tools=[retrieve_context],
      checkpointer=agent_memory
)

#query
while True:
      query = input("Enter your query")
      if query==quit:
            break
      
      response = agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":1}}
      )
      answer = response["messages"][-1].content
      print(answer)

      