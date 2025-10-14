#!/usr/bin/env python3
"""
Setup script for Ask My Docs
Helps users install dependencies and configure the environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run("ollama --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
        else:
            print("❌ Ollama is not installed")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False

def check_redis():
    """Check if Redis is running."""
    try:
        result = subprocess.run("redis-cli -p 6380 ping", shell=True, capture_output=True, text=True)
        if result.returncode == 0 and "PONG" in result.stdout:
            print("✅ Redis is running on port 6380")
            return True
        else:
            print("❌ Redis is not running on port 6380")
            return False
    except FileNotFoundError:
        print("❌ Redis is not installed")
        return False

def install_dependencies():
    """Install Python dependencies."""
    # Try pip3 first, then pip
    if run_command("pip3 install -r requirements.txt", "Installing Python dependencies"):
        return True
    else:
        return run_command("pip install -r requirements.txt", "Installing Python dependencies (fallback)")

def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("🔄 Creating virtual environment...")
        if run_command("python3 -m venv venv", "Creating virtual environment"):
            print("✅ Virtual environment created")
            print("📝 To activate the virtual environment:")
            print("   source venv/bin/activate  # On macOS/Linux")
            print("   venv\\Scripts\\activate     # On Windows")
            return True
        else:
            return False
    else:
        print("✅ Virtual environment already exists")
        return True

def download_model():
    """Download a default model."""
    print("🔄 Downloading llama3 model (this may take a while)...")
    return run_command("ollama pull llama3", "Downloading llama3 model")

def main():
    """Main setup function."""
    print("🚀 Setting up Ask My Docs...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        print("\n📝 To install Ollama:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        print("   Then restart your terminal and run this script again")
        sys.exit(1)
    
    # Check Redis
    if not check_redis():
        print("\n📝 To start Redis:")
        print("   macOS: brew install redis && redis-server")
        print("   Ubuntu: sudo apt-get install redis-server && redis-server")
        print("   Or use Docker: docker run -d -p 6379:6379 redis:alpine")
        print("   Then run this script again")
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Activate virtual environment and install dependencies
    print("📝 Activating virtual environment and installing dependencies...")
    print("   Please run the following commands manually:")
    print("   source venv/bin/activate")
    print("   pip install -r requirements.txt")
    print("   Or if using Windows:")
    print("   venv\\Scripts\\activate")
    print("   pip install -r requirements.txt")
    
    # Try to install dependencies anyway
    install_dependencies()
    
    # Download model
    if not download_model():
        print("⚠️  Model download failed, but you can download it manually later:")
        print("   ollama pull llama3")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Activate virtual environment:")
    print("   source venv/bin/activate  # On macOS/Linux")
    print("   venv\\Scripts\\activate     # On Windows")
    print("\n2. Add your documents to the 'documents' folder")
    print("\n3. Run the application:")
    print("   streamlit run app.py")
    print("\n4. Open your browser to http://localhost:8501")
    print("\n📚 Happy Document Chatting!")

if __name__ == "__main__":
    main() 