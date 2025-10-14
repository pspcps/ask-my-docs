#!/usr/bin/env python3
"""
Test script for Ask My Docs
Verifies that all components are working correctly.
"""

import sys
import subprocess
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("🔄 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import ollama
        print("✅ Ollama imported successfully")
    except ImportError as e:
        print(f"❌ Ollama import failed: {e}")
        return False
    
    try:
        import redis
        print("✅ Redis imported successfully")
    except ImportError as e:
        print(f"❌ Redis import failed: {e}")
        return False
    
    try:
        from langchain.vectorstores import Redis
        from langchain.embeddings import OllamaEmbeddings
        from langchain.llms import Ollama
        print("✅ LangChain imports successful")
    except ImportError as e:
        print(f"❌ LangChain import failed: {e}")
        return False
    
    try:
        from data_ingest import DocumentProcessor
        print("✅ DocumentProcessor imported successfully")
    except ImportError as e:
        print(f"❌ DocumentProcessor import failed: {e}")
        return False
    
    return True

def test_ollama():
    """Test if Ollama is running."""
    print("🔄 Testing Ollama connection...")
    
    try:
        import ollama
        models = ollama.list()
        print("✅ Ollama is running")
        if 'models' in models and models['models']:
            model_names = [m.get('name', 'unknown') for m in models['models']]
            print(f"   Available models: {model_names}")
        else:
            print("   No models found")
        return True
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("   Please start Ollama: ollama serve")
        return False

def test_chromadb():
    """Test if ChromaDB is available."""
    print("🔄 Testing ChromaDB availability...")
    
    try:
        import chromadb
        print("✅ ChromaDB is available")
        return True
    except Exception as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False

def test_documents():
    """Test document processing."""
    print("🔄 Testing document processing...")
    
    try:
        from data_ingest import DocumentProcessor
        processor = DocumentProcessor()
        
        docs_path = Path("documents")
        if not docs_path.exists():
            print("⚠️  Documents directory not found")
            return False
        
        # Test document info
        info = processor.get_document_info(docs_path)
        print(f"✅ Found {info.get('total_files', 0)} documents")
        
        # Test document loading
        documents = processor.load_documents(docs_path)
        print(f"✅ Loaded {len(documents)} document chunks")
        
        if documents:
            print(f"   Sample document: {documents[0].metadata.get('file_name', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Document processing failed: {e}")
        return False

def test_app_components():
    """Test main app components."""
    print("🔄 Testing app components...")
    
    try:
        from app import AskMyDocs
        app = AskMyDocs()
        print("✅ AskMyDocs class initialized")
        return True
    except Exception as e:
        print(f"❌ App component test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running Ask My Docs Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Ollama Test", test_ollama),
        ("ChromaDB Test", test_chromadb),
        ("Document Test", test_documents),
        ("App Component Test", test_app_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n📝 Next steps:")
        print("1. Run: streamlit run app.py")
        print("2. Open: http://localhost:8501")
        print("3. Click 'Process Documents' in the sidebar")
        print("4. Start asking questions!")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\n🔧 Common fixes:")
        print("- Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("- Start Ollama: ollama serve")
        print("- Install Redis: brew install redis (macOS) or sudo apt-get install redis-server (Ubuntu)")
        print("- Start Redis: redis-server")
        print("- Install dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 