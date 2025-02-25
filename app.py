import streamlit as st
import os
import faiss
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def process_pdfs(pdf_files):
    """Extracts text from uploaded PDFs and returns document chunks."""
    docs = []
    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(pdf_file)
        pages = loader.load()
        docs.extend(pages)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(docs)

def create_vector_store(chunks):
    """Creates FAISS vector store from document chunks."""
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("test")))
    vector_store = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    vector_store.add_documents(chunks)
    return vector_store

def build_rag_pipeline(vector_store):
    """Creates a RAG pipeline with retrieval and LLM answering."""
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 100, 'lambda_mult': 1})
    model = ChatOllama(model="llama3.2:1b", base_url="http://localhost:11434")
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an assistant for question-answering tasks. Use the retrieved context to answer.
        If you don't know, say you don't know. Answer in bullet points.
        Question: {question}
        Context: {context}
        Answer:
        """
    )
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    return ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt_template | model | StrOutputParser())

# Streamlit UI
st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")
st.title("📄 Chat with PDF using LLM")
st.markdown("Chat with your PDF documents interactively. Upload PDFs and ask questions.")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.success("Processing PDFs...")
    pdf_paths = [os.path.join("temp", file.name) for file in uploaded_files]
    os.makedirs("temp", exist_ok=True)
    for file, path in zip(uploaded_files, pdf_paths):
        with open(path, "wb") as f:
            f.write(file.read())
    
    chunks = process_pdfs(pdf_paths)
    vector_store = create_vector_store(chunks)
    rag_chain = build_rag_pipeline(vector_store)
    
    st.subheader("Chat with your PDFs")
    chat_history = st.session_state.get("chat_history", [])
    
    for entry in chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["message"])
    
    question = st.chat_input("Ask a question about the PDFs")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        
        output = rag_chain.invoke(question)
        with st.chat_message("assistant"):
            st.markdown(output)
        
        chat_history.append({"role": "user", "message": question})
        chat_history.append({"role": "assistant", "message": output})
        st.session_state.chat_history = chat_history
