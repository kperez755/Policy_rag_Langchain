# Policy RAG with LangChain

This repository contains a Retrieval-Augmented Generation (RAG) application built with FastAPI, LangChain, LangGraph, and ChromaDB.  
The system ingests policy documents (PDFs), stores them in a vector database, and enables semantic search and question answering over those documents using OpenAI models.

---

## Features

- FastAPI backend
- PDF ingestion and text chunking
- Semantic search with ChromaDB
- Retrieval-Augmented Generation (RAG)
- LangChain + LangGraph orchestration
- OpenAI-powered responses
- In-memory checkpointing for graph state

---

## Tech Stack

- Python
- FastAPI
- LangChain
- LangGraph
- OpenAI API
- ChromaDB
- PyPDF
- Uvicorn

---

## Project Structure (High-Level)

```
.
├── Static
  ├── index.html           # contains html for frontend
├── server.py                # FastAPI app entry point
├── WithSemantic.py        # Ingestion + retrieval logic
├── requirements.txt
├── README.md
└── .env                   # Not committed (you must create this)
```

---

## Environment Variables

You must create your own `.env` file in the project root.  
This file is required for the application to run and is not included in the repository.

Create a file named `.env` and add the following variables:

```env
OPENAI_API_KEY=""

CHROMA_API_KEY=""
CHROMA_TENANT=""
CHROMA_DATABASE=""
CHROMA_COLLECTION_NAME=""
```

Fill in each value with your own credentials.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/kperez755/Policy_rag_Langchain.git
cd Policy_rag_Langchain
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the FastAPI server with Uvicorn:

```bash
uvicorn main:app --reload
```

By default, the API will be available at:

```
http://localhost:8000
```

Interactive API documentation is available at:

```
http://localhost:8000/docs
```

---

## Usage Overview

1. Ingest PDF documents into ChromaDB
2. Documents are split into chunks and embedded
3. Queries are semantically matched against stored chunks
4. Relevant context is passed to the language model
5. The model generates responses grounded in retrieved content

---

## Notes

- A `.env` file is required and must be created manually
- Do not commit API keys or credentials to source control
- This project is intended for experimentation and learning with RAG systems
