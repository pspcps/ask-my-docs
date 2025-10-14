#!/bin/bash

echo "🚀 Setting up Ask My Docs..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "🔄 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "🔄 Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    echo "📝 Try running manually:"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Check if Ollama is running
echo "🔄 Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
else
    echo "❌ Ollama is not installed"
    echo "📝 Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Check if Redis is running on port 6380
echo "🔄 Checking Redis..."
if redis-cli -p 6380 ping &> /dev/null; then
    echo "✅ Redis is running on port 6380"
else
    echo "❌ Redis is not running on port 6380"
    echo "📝 Start Redis: redis-server --port 6380"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Add your documents to the 'documents' folder"
echo "3. Run the application: streamlit run app.py"
echo "4. Open your browser to http://localhost:8501"
echo ""
echo "📚 Happy Document Chatting!" 