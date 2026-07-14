from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver 


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def processing_document(path):

    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = InMemoryVectorStore.from_documents(
    embedding=embeddings,
    documents=texts
    )

    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query:str):
        """Retrieve documents relevant to a query from the knowledge base."""
        context = ""
        docs = vector_store.similarity_search(query=query, k=3)
        for doc in docs:
            context += doc.page_content + "\n\n"

        return context
    
    system_prompt = """You are a helpful assistant that answers questions using retrieved context. 
        My knowledge base consists of the details from the uploaded document. 
        ALWAYS use the `retrieve_context` tool for questions requiring external knowledge."""
    
    agent_memory = InMemorySaver()

    agent = create_agent(
      system_prompt=system_prompt,
      model=llm,
      tools=[retrieve_context],
      checkpointer=agent_memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True





if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF Files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "./data/uploads"
            for file in uploaded:
                with open(os.path.join(path, file.name), "wb") as f:
                    f.write(file.getvalue())

            processing_document(path)
            st.rerun()

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)


query = st.chat_input("Ask anything related to uploaded documents....")
if query:
        st.session_state.messages.append({"role":"user", "content":query})

        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai", "content":answer})   