#!/usr/bin/env python3
"""
Ask My Docs - A Local AI Chatbot for Your PDFs or Data
Main application script that provides a chat interface for querying documents.
"""

import os
import sys
from pathlib import Path
import streamlit as st
from data_ingest import DocumentProcessor
import ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import requests

# Configuration
COLLECTION_NAME = "ask_my_docs"
MODEL_NAME = "llama3"  # or "mistral", "codellama", etc.

# Ensure Ollama host is explicitly set for reliability in various shells
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

class AskMyDocs:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=MODEL_NAME)
        self.llm = Ollama(model=MODEL_NAME)
        self.vector_store = None
        self.qa_chain = None
        
    def setup_vector_store(self):
        """Initialize or connect to the vector store"""
        try:
            # Try to load existing vector store
            if os.path.exists("chroma_db"):
                self.vector_store = Chroma(
                    persist_directory="chroma_db",
                    embedding_function=self.embeddings,
                    collection_name=COLLECTION_NAME
                )
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error connecting to vector store: {e}")
            return False
    
    def process_documents(self, documents_path="documents"):
        """Process documents and create embeddings"""
        processor = DocumentProcessor()
        docs_path = Path(documents_path)
        
        if not docs_path.exists():
            st.warning(f"Documents directory '{documents_path}' not found. Please add some PDFs or text files.")
            return False
        
        with st.spinner("Processing documents..."):
            try:
                # Process documents
                documents = processor.load_documents(docs_path)
                if not documents:
                    st.warning("No documents found to process.")
                    return False
                
                # Create vector store
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory="chroma_db",
                    collection_name=COLLECTION_NAME
                )
                
                # Persist the vector store
                self.vector_store.persist()
                
                st.success(f"Successfully processed {len(documents)} documents!")
                return True
                
            except Exception as e:
                st.error(f"Error processing documents: {e}")
                return False
    
    def setup_qa_chain(self):
        """Setup the question-answering chain"""
        if not self.vector_store:
            return False
            
        # Create a custom prompt template
        template = """You are a helpful AI assistant that answers questions based on the provided context.
        
        Context: {context}
        
        Question: {question}
        
        Answer the question based on the context above. If the answer cannot be found in the context, say "I don't have enough information to answer this question."
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
        return True
    
    def ask_question(self, question):
        """Ask a question and get an answer"""
        if not self.qa_chain:
            return "Please process documents first."
        
        try:
            with st.spinner("Thinking..."):
                answer = self.qa_chain.run(question)
                return answer
        except Exception as e:
            return f"Error getting answer: {e}"

def main():
    st.set_page_config(
        page_title="Ask My Docs",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Ask My Docs")
    st.markdown("**Your private AI chatbot for your documents**")
    
    # Initialize the app
    if 'ask_my_docs' not in st.session_state:
        st.session_state.ask_my_docs = AskMyDocs()
    
    app = st.session_state.ask_my_docs
    
    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Robust Ollama check via direct HTTP ping
        try:
            base = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
            resp = requests.get(f"{base}/api/tags", timeout=5)
            if resp.status_code == 200:
                tags = resp.json()
                model_names = [m.get('name', 'unknown') for m in tags.get('models', [])]
                st.success("✅ Ollama is running")
                st.write("Available models:", model_names or ["(none)"])
            else:
                st.error("❌ Ollama is not reachable")
                st.caption(f"HTTP {resp.status_code} from {base}/api/tags")
                st.info("Please start Ollama first: `ollama serve`")
                return
        except Exception as e:
            st.error("❌ Ollama is not running")
            st.caption(f"Detail: {e}")
            st.info("Please start Ollama first: `ollama serve`")
            return
        
        st.divider()
        
        # Document processing
        if st.button("🔄 Process Documents", type="primary"):
            if app.process_documents():
                app.setup_qa_chain()
                st.success("Documents processed successfully!")
    
    # Main chat interface
    if app.qa_chain:
        st.success("✅ Ready to answer questions!")
        
        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                response = app.ask_question(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    else:
        st.info("👆 Click 'Process Documents' in the sidebar to get started!")
        
        # Show sample questions
        st.subheader("💡 Example Questions")
        st.markdown("""
        Once you've processed your documents, you can ask questions like:
        
        - "What is the company's leave policy?"
        - "How does the refund process work?"
        - "What are the main features of the product?"
        - "What are the requirements for the project?"
        """)

if __name__ == "__main__":
    main() 