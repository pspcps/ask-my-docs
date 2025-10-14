#!/usr/bin/env python3
"""
Flask API for Ask My Docs
Provides a REST API interface for the document chatbot.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
from data_ingest import DocumentProcessor
import ollama
import redis
from langchain.vectorstores import Redis
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
REDIS_URL = "redis://localhost:6380"
COLLECTION_NAME = "ask_my_docs"
MODEL_NAME = "llama3"

class AskMyDocsAPI:
    def __init__(self):
        self.redis_client = redis.from_url(REDIS_URL)
        self.embeddings = OllamaEmbeddings(model=MODEL_NAME)
        self.llm = Ollama(model=MODEL_NAME)
        self.vector_store = None
        self.qa_chain = None
        
    def setup_vector_store(self):
        """Initialize or connect to the vector store"""
        try:
            self.vector_store = Redis.from_existing_index(
                embedding=self.embeddings,
                index_name=COLLECTION_NAME,
                redis_url=REDIS_URL
            )
            return True
        except Exception as e:
            return False
    
    def process_documents(self, documents_path="documents"):
        """Process documents and create embeddings"""
        processor = DocumentProcessor()
        docs_path = Path(documents_path)
        
        if not docs_path.exists():
            return False, "Documents directory not found"
        
        try:
            # Process documents
            documents = processor.load_documents(docs_path)
            if not documents:
                return False, "No documents found to process"
            
            # Create vector store
            self.vector_store = Redis.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=COLLECTION_NAME,
                redis_url=REDIS_URL
            )
            
            return True, f"Successfully processed {len(documents)} documents"
            
        except Exception as e:
            return False, f"Error processing documents: {e}"
    
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
            return "Please process documents first"
        
        try:
            answer = self.qa_chain.run(question)
            return answer
        except Exception as e:
            return f"Error getting answer: {e}"

# Initialize the API instance
api_instance = AskMyDocsAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check Ollama
        models = ollama.list()
        ollama_status = "running"
    except:
        ollama_status = "not running"
    
    try:
        # Check Redis
        api_instance.redis_client.ping()
        redis_status = "connected"
    except:
        redis_status = "not connected"
    
    return jsonify({
        "status": "ok",
        "ollama": ollama_status,
        "redis": redis_status
    })

@app.route('/process-documents', methods=['POST'])
def process_documents():
    """Process documents endpoint"""
    try:
        success, message = api_instance.process_documents()
        if success:
            api_instance.setup_qa_chain()
            return jsonify({
                "success": True,
                "message": message
            })
        else:
            return jsonify({
                "success": False,
                "message": message
            }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {e}"
        }), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question endpoint"""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                "success": False,
                "message": "Question is required"
            }), 400
        
        question = data['question']
        answer = api_instance.ask_question(question)
        
        return jsonify({
            "success": True,
            "question": question,
            "answer": answer
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {e}"
        }), 500

@app.route('/documents', methods=['GET'])
def list_documents():
    """List available documents"""
    try:
        docs_path = Path("documents")
        if not docs_path.exists():
            return jsonify({
                "success": True,
                "documents": [],
                "message": "Documents directory not found"
            })
        
        documents = []
        for file_path in docs_path.iterdir():
            if file_path.is_file():
                documents.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "type": file_path.suffix
                })
        
        return jsonify({
            "success": True,
            "documents": documents
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {e}"
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Get application status"""
    try:
        # Check if QA chain is ready
        qa_ready = api_instance.qa_chain is not None
        
        # Check vector store
        vector_store_ready = api_instance.vector_store is not None
        
        return jsonify({
            "success": True,
            "qa_chain_ready": qa_ready,
            "vector_store_ready": vector_store_ready,
            "model": MODEL_NAME,
            "collection": COLLECTION_NAME
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {e}"
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Ask My Docs API...")
    print("📝 API Endpoints:")
    print("   GET  /health              - Health check")
    print("   POST /process-documents   - Process documents")
    print("   POST /ask                 - Ask a question")
    print("   GET  /documents           - List documents")
    print("   GET  /status              - Get status")
    print("\n🌐 API will be available at http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 