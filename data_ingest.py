#!/usr/bin/env python3
"""
Data Ingestion Module for Ask My Docs
Handles loading and processing of PDFs, text files, and other documents.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and processing for the Ask My Docs application."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Supported file extensions and their corresponding loaders
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.docx': UnstructuredWordDocumentLoader,
            '.doc': UnstructuredWordDocumentLoader,
            '.csv': CSVLoader
        }
    
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return list(self.loaders.keys())
    
    def is_supported_file(self, file_path: Path) -> bool:
        """Check if a file is supported based on its extension."""
        return file_path.suffix.lower() in self.loaders
    
    def load_single_document(self, file_path: Path) -> Optional[List[Document]]:
        """Load a single document and return a list of Document objects."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return None
            
            if not self.is_supported_file(file_path):
                logger.warning(f"Unsupported file type: {file_path}")
                return None
            
            # Get the appropriate loader
            loader_class = self.loaders[file_path.suffix.lower()]
            
            # Initialize loader with file path
            if file_path.suffix.lower() == '.csv':
                loader = loader_class(str(file_path), encoding='utf-8')
            else:
                loader = loader_class(str(file_path))
            
            # Load documents
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': file_path.suffix.lower(),
                    'file_name': file_path.name
                })
            
            logger.info(f"Successfully loaded {len(documents)} pages/chunks from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None
    
    def load_documents(self, directory_path: Path) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of Document objects
        """
        documents = []
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return documents
        
        if not directory_path.is_dir():
            logger.error(f"Path is not a directory: {directory_path}")
            return documents
        
        # Find all files in the directory
        files = []
        for ext in self.get_supported_extensions():
            files.extend(directory_path.glob(f"*{ext}"))
            files.extend(directory_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(files)} supported files in {directory_path}")
        
        # Load each document
        for file_path in files:
            doc_chunks = self.load_single_document(file_path)
            if doc_chunks:
                documents.extend(doc_chunks)
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better processing.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of split Document objects
        """
        if not documents:
            return []
        
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            return documents
    
    def process_documents(self, directory_path: Path) -> List[Document]:
        """
        Complete document processing pipeline: load and split documents.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of processed Document objects ready for embedding
        """
        # Load documents
        documents = self.load_documents(directory_path)
        
        if not documents:
            logger.warning("No documents found to process")
            return []
        
        # Split documents into chunks
        processed_docs = self.split_documents(documents)
        
        logger.info(f"Processing complete. {len(processed_docs)} chunks ready for embedding.")
        return processed_docs
    
    def get_document_info(self, directory_path: Path) -> dict:
        """
        Get information about documents in a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Dictionary with document statistics
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists() or not directory_path.is_dir():
            return {"error": "Directory not found"}
        
        files = []
        for ext in self.get_supported_extensions():
            files.extend(directory_path.glob(f"*{ext}"))
            files.extend(directory_path.glob(f"*{ext.upper()}"))
        
        file_types = {}
        for file_path in files:
            ext = file_path.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            "total_files": len(files),
            "supported_files": len(files),
            "file_types": file_types,
            "supported_extensions": self.get_supported_extensions()
        }

def main():
    """Test the document processor."""
    processor = DocumentProcessor()
    
    # Test with current directory
    docs_path = Path("documents")
    
    if docs_path.exists():
        print("Document Information:")
        info = processor.get_document_info(docs_path)
        print(f"  Total files: {info.get('total_files', 0)}")
        print(f"  File types: {info.get('file_types', {})}")
        
        print("\nProcessing documents...")
        documents = processor.process_documents(docs_path)
        print(f"  Processed {len(documents)} document chunks")
        
        if documents:
            print("\nSample document metadata:")
            sample_doc = documents[0]
            print(f"  Source: {sample_doc.metadata.get('source', 'N/A')}")
            print(f"  File type: {sample_doc.metadata.get('file_type', 'N/A')}")
            print(f"  Content length: {len(sample_doc.page_content)} characters")
    else:
        print(f"Documents directory '{docs_path}' not found.")
        print("Please create a 'documents' directory and add some PDFs or text files.")

if __name__ == "__main__":
    main() 