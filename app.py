import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import time

# Load API key
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM (Groq ka model, not embedding model)
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")

# Prompt Template
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Please provide the most accurate response based on the question.

<context>
{context}
</context>
Question: {input}
""")

# Function to create vector embeddings
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        # Use Ollama embedding model
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Load PDFs
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()

        # Split into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])

        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, 
            st.session_state.embeddings
        )

st.title("RAG Documents Q&A with Groq LLM")

# UI Input
user_prompt = st.text_input("Enter your query from the research paper")

# Button for embedding
if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector Database is ready")

# If query is entered
if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    st.write(f"Response Time: {time.process_time() - start} seconds")

    # Show answer
    st.write(response["answer"])

    # Show similar documents
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write('------------------------')
