# RAG-Documents-Q-A-with-Groq-ollama
# RAG Documents Q&A with Groq LLM

This project is a **Retrieval-Augmented Generation (RAG)** application built using **LangChain**, **Groq LLM**, and **Ollama Embeddings**.  
It allows you to upload and query research papers (PDFs) and get accurate answers using a combination of **vector search (FAISS)** and **large language models (LLMs)**.

---

## **Features**
- Load and process multiple PDFs from a folder.
- Split documents into manageable chunks for semantic search.
- Generate embeddings locally using `Ollama` (`nomic-embed-text` model).
- Store embeddings in `FAISS` vector database for fast retrieval.
- Query documents using `Groq LLM` (`Gemma-7b-It`) for natural language answers.
- Display relevant document context (similarity search results) with the answers.
- Built with `Streamlit` for an interactive web UI.

---

## **Project Structure**

RAG DOCUMENT Q&A/
│
├── research_papers/ # Folder containing your PDF documents
├── app.py # Main Streamlit application
├── .env # Environment variables (API keys)
└── README.md # Project documentation
