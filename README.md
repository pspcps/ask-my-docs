# 📚 Ask My Docs

A **private, local AI chatbot** that can answer questions from your PDFs, text files, or personal notes using **Ollama + LangChain + Redis Vector Database**.

🔒 **100% Private & Local** - Your data never leaves your machine!

## 🎯 What You'll Learn

| Step | Concept | Description |
|------|---------|-------------|
| 1 | Setup | Installing Ollama, Python, environment |
| 2 | Load Model | Running a local LLM (llama3, mistral, etc.) |
| 3 | Data Ingestion | Loading PDFs/text using LangChain |
| 4 | Embeddings | Turning text into vectors using ollama.embeddings |
| 5 | Vector Database | Storing & retrieving relevant info using Redis |
| 6 | RAG | Connecting retrieved info with Ollama for answering |
| 7 | Chat Interface | Beautiful UI using Streamlit |
| 8 | API (optional) | Expose your chatbot via Flask API |

## 🛠️ Tools Used

| Tool | Purpose |
|------|---------|
| **Ollama** | Runs local LLM like llama3 |
| **LangChain** | Framework for RAG and data handling |
| **Redis** | Vector store to retrieve info |
| **Python** | Main language |
| **Streamlit** | Beautiful web interface |

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** (for local LLMs):
   ```bash
   # macOS
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows
   # Download from https://ollama.ai/download
   ```

2. **Install Redis** (for vector database):
   ```bash
   # macOS
   brew install redis
   
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # Or use Docker
   docker run -d -p 6380:6379 redis:alpine
   ```

3. **Install Python 3.8+**:
   ```bash
   # Check your version
   python3 --version
   ```

### Installation

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd ask-my-docs
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Start services**:
   ```bash
   # Start Ollama (in one terminal)
   ollama serve
   
   # Start Redis (in another terminal)
   redis-server
   
   # Or if using Docker for Redis:
   docker run -d -p 6379:6379 redis:alpine
   ```

3. **Download a model**:
   ```bash
   # Download llama3 (recommended)
   ollama pull llama3
   
   # Or try other models
   ollama pull mistral
   ollama pull codellama
   ```

4. **Add your documents**:
   ```bash
   # Create documents directory
   mkdir documents
   
   # Add your PDFs, text files, etc.
   cp your-document.pdf documents/
   cp your-notes.txt documents/
   ```

5. **Run the app**:
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** and go to `http://localhost:8501`

## 📁 Project Structure

```
ask-my-docs/
├── app.py                  # Main Streamlit application
├── data_ingest.py          # Document loading & processing
├── documents/              # Your PDFs, text files, etc.
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎯 Example Use Cases

### Company Documentation
Upload your company's documentation and ask:
- "How does the leave policy work?"
- "What is the refund process?"
- "What are the main features of our product?"

### Research Papers
Upload research papers and ask:
- "What are the main findings?"
- "What methodology was used?"
- "What are the limitations mentioned?"

### Personal Notes
Upload your notes and ask:
- "What did I write about machine learning?"
- "What are my project ideas?"
- "What are my meeting notes from last week?"

## 🔧 Configuration

### Supported File Types
- **PDF** (.pdf)
- **Text** (.txt)
- **Markdown** (.md)
- **Word Documents** (.docx, .doc)
- **CSV** (.csv)

### Model Options
You can change the model in `app.py`:
```python
MODEL_NAME = "llama3"  # or "mistral", "codellama", etc.
```

### Vector Database
The app uses Redis as the vector database. You can change the configuration:
```python
REDIS_URL = "redis://localhost:6379"
COLLECTION_NAME = "ask_my_docs"
```

## 🚀 Advanced Features

### Custom Prompts
Modify the prompt template in `app.py` to customize how the AI responds:
```python
template = """You are a helpful AI assistant that answers questions based on the provided context.
...
```

### Document Chunking
Adjust chunk size and overlap in `data_ingest.py`:
```python
chunk_size=1000,      # Size of each text chunk
chunk_overlap=200     # Overlap between chunks
```

### Multiple Models
You can easily switch between different Ollama models:
```bash
# Download different models
ollama pull llama3
ollama pull mistral
ollama pull codellama
ollama pull neural-chat
```

## 🐛 Troubleshooting

### Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve
```

### Redis Issues
```bash
# Check if Redis is running on port 6380
redis-cli -p 6380 ping

# Start Redis manually on port 6380
redis-server --port 6380
```

### Python Dependencies
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python3 --version
```

### Memory Issues
If you're running out of memory:
1. Use a smaller model: `ollama pull llama3:3b`
2. Reduce chunk size in `data_ingest.py`
3. Close other applications

## 🔒 Privacy & Security

- **100% Local**: All processing happens on your machine
- **No Data Sent**: Your documents never leave your computer
- **Open Source**: Full transparency of the code
- **No API Keys**: No external services required

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [LangChain](https://langchain.com/) for the RAG framework
- [Streamlit](https://streamlit.io/) for the web interface
- [Redis](https://redis.io/) for vector storage

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Search existing issues
3. Create a new issue with details

---

**Happy Document Chatting! 🎉** 