from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st

### data in st.session

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None;

if "messages" not in st.session_state:
    st.session_state.messages = [];


def process_document(path):
    ## Load Document
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ## Split the doc
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    splitted_docs = splitter.split_documents(docs)

    ## Vector Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

    ## Store in DB
    vector_db = InMemoryVectorStore.from_documents(
        documents=splitted_docs,
        embedding=embeddings
    )

    doc = vector_db.similarity_search("Patient name")
    print(doc[0])


    ## Agent - Tool, LLm, Prompt

    llm = ChatGroq(model="openai/gpt-oss-20b")


    @tool
    def retriever_tool(query:str):
        """
        This tool can help you to retrive the relevant data of the PDF Documents, and these documents 
        have details about the medical reports
        """
        print("Tool called", query)
        context = ""

        docs = vector_db.similarity_search(query=query, k = 4)

        for doc in docs:
            context += doc.page_content + "\n\n"

        return context


    system_prompt = """
        You are a PDF question-answering assistant.
        
        The user has uploaded PDF documents containing medical report information.
        
        For ANY question that could be answered from the uploaded PDF,
        you MUST call `retriever_tool` before answering.
        
        Never ask the user to provide the document or additional context.
        Use `retriever_tool` to retrieve the relevant information first.
        
        After receiving the tool result, answer the user's question using
        only the retrieved information.
    """

    memory = InMemorySaver()

    agent = create_agent(
        model = llm,
        tools = [retriever_tool],
        system_prompt = system_prompt,
        checkpointer = memory
    )  
    st.session_state.agent = agent
    st.session_state.document_uploaded = True

### upload UI

if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF files", type=["pdf"], accept_multiple_files= True)
    if uploaded:
        with st.spinner("Processing..."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f :
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()


### Chat UI
if st.session_state.document_uploaded and st.session_state.agent:
    query = st.chat_input("Ask Anything from the Uploaded Documents..")
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user","content":query}]},
            {"configurable":{"thread_id":"1"}}
        )                  

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai","content":answer})
