# Chat with PDF using Streamlit and FAISS

## Overview
This project enables users to upload PDF documents and interactively chat with them using an LLM-powered Retrieval-Augmented Generation (RAG) pipeline. The system processes PDFs, extracts text, creates vector embeddings using FAISS, and retrieves relevant information to answer user queries.

## Features
- Upload and process multiple PDFs.
- Extract text from PDFs using `PyMuPDFLoader`.
- Chunk and embed text using `RecursiveCharacterTextSplitter` and `OllamaEmbeddings`.
- Store embeddings in a FAISS vector database for efficient retrieval.
- Use `ChatOllama` for natural language responses.
- Interactive chat interface with chat history support using `Streamlit`.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip
- FAISS
- Streamlit
- LangChain
- Ollama LLM Server

### Install Dependencies
```sh
pip install streamlit faiss-cpu langchain-community langchain-core langchain-ollama
```

## Running the Application

### Start the Ollama LLM server
Ensure it's running on `http://localhost:11434`.

### Run the Streamlit app
```sh
streamlit run app.py
```

Open the browser at `http://localhost:8501`.

## Usage
1. Upload one or more PDF files.
2. Wait for the system to process the documents.
3. Ask questions related to the uploaded PDFs.
4. The assistant retrieves relevant context and provides an answer.

## File Structure
```
├── app.py            # Main Streamlit app
├── requirements.txt  # List of dependencies
├── temp/             # Temporary storage for uploaded PDFs
└── README.md         # Documentation
```

## Notes
- Ensure Ollama is running before starting the app.
- FAISS is used for fast document retrieval.
- Modify chunk size and overlap in `RecursiveCharacterTextSplitter` for better performance.

## License
This project is licensed under the MIT License.

