# StudyHub - Unified AI Learning Platform

StudyHub is a comprehensive web application that brings together three powerful AI tools under one roof. Built with Flask, it provides a seamless experience for AI-powered learning and content analysis.

## 🚀 Features

### 🌟 Unified Interface
- **Single Application**: All tools in one place at `localhost:5000`
- **Modern UI**: Bootstrap-based responsive design
- **Seamless Navigation**: Easy switching between tools
- **Real-time Status**: Live API and service monitoring

### 📚 PDF Chat
- Upload and chat with multiple PDF documents
- AI-powered Q&A using Ollama (Mistral model)
- Vector-based document search with FAISS
- Persistent chat sessions

### 🎬 Content Summarizer  
- YouTube video analysis and summarization
- Webpage content extraction and analysis
- Text content summarization
- Powered by Google Gemini API

### ❓ Quiz Generator
- Generate quizzes from PDF documents
- Multiple question types (Multiple Choice, True/False, Essay)
- AI-powered question generation
- Export functionality

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- [Ollama](https://ollama.ai) (for PDF chat functionality)
- Google Gemini API key (for summarizer and quiz generator)

### Quick Start

1. **Clone/Download and Navigate**
   ```bash
   cd StudyHub
   ```

2. **Run the Setup Script (Recommended)**
   ```bash
   start_studyhub.bat
   ```
   
   This script will:
   - Create a virtual environment
   - Install all dependencies
   - Set up configuration files
   - Check for Ollama
   - Start the application

3. **Manual Setup (Alternative)**
   ```bash
   # Create virtual environment
   python -m venv venv
   venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Copy environment file
   copy .env.example .env
   
   # Edit .env with your API keys
   notepad .env
   
   # Start application
   python app.py
   ```

### Configuration

1. **Edit the `.env` file** with your API keys:
   ```
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```

2. **Install and Configure Ollama** (for PDF Chat):
   ```bash
   # Download from https://ollama.ai
   # Then run:
   ollama pull mistral
   ```

## 🌐 Usage

Once running, access StudyHub at: **http://localhost:5000**

### Dashboard
- Overview of all available tools
- Real-time service status monitoring
- Quick access to all features

### PDF Chat (`/pdf-chat`)
1. Upload one or more PDF files
2. Wait for processing (creates vector embeddings)
3. Start asking questions about your documents
4. Use quick question buttons or type custom queries

### Summarizer (`/summarizer`)
1. Select content type (YouTube, Webpage, or Text)
2. Enter your URL or paste text content
3. Choose summary type (Concise or Detailed)
4. Generate AI-powered summaries

### Quiz Generator (`/quiz-generator`)
1. Upload PDF or document files
2. Select question types and quantity
3. Generate AI-created quiz questions
4. Download or copy the generated quiz

## 🏗️ Architecture

### Technology Stack
- **Backend**: Flask (Python web framework)
- **Frontend**: Bootstrap 5, vanilla JavaScript
- **AI/ML**: 
  - Google Gemini API (summarization, quiz generation)
  - Ollama + Mistral (local LLM for PDF chat)
  - LangChain (document processing)
  - FAISS (vector similarity search)
  - HuggingFace Transformers (embeddings)

### File Structure
```
StudyHub/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── start_studyhub.bat    # Windows launcher script
├── .env.example          # Environment template
├── templates/            # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── dashboard.html    # Main dashboard
│   ├── pdf_chat.html     # PDF chat interface
│   ├── summarizer.html   # Content summarizer
│   ├── quiz_generator.html # Quiz generator
│   └── error.html        # Error pages
├── static/               # Static assets
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── main.js       # JavaScript functionality
├── uploads/              # Temporary file uploads
└── vector_stores/        # FAISS vector databases
```

## 🔧 Configuration Options

### Environment Variables (`.env`)
- `GEMINI_API_KEY`: Google Gemini API key for AI features
- `GEMINI_MODEL`: Model to use (default: models/gemini-2.5-flash)
- `FLASK_ENV`: Development/production mode
- `SECRET_KEY`: Flask session security key
- `MAX_CONTENT_LENGTH`: Maximum file upload size
- `OLLAMA_HOST`: Ollama service URL (default: http://localhost:11434)

### Application Settings
- **Upload Limits**: 50MB per file by default
- **Session Management**: Server-side session storage
- **Vector Stores**: Persistent FAISS indexes per session
- **API Timeouts**: Configurable request timeouts

## 🚨 Troubleshooting

### Common Issues

**1. PDF Chat Not Working**
- Ensure Ollama is installed and running
- Check if Mistral model is downloaded: `ollama list`
- Verify Ollama is accessible at `http://localhost:11434`

**2. Summarizer/Quiz Generator Errors**
- Verify Gemini API key in `.env` file
- Check internet connection
- Ensure API quota is available

**3. File Upload Issues**
- Check file size (50MB limit)
- Ensure PDF files are text-based (not scanned images)
- Verify upload directory permissions

**4. Application Won't Start**
- Check Python version (3.8+ required)
- Verify all dependencies installed: `pip list`
- Check for port conflicts (port 5000)

### Getting Help
- Check the browser console for JavaScript errors
- Review the terminal output for Python errors
- Verify all services are running (API status in dashboard)

## 🔒 Security & Privacy

- **Local Processing**: PDF content is processed locally
- **Session Isolation**: Each user session has isolated vector stores
- **Temporary Storage**: Uploaded files are deleted after processing
- **API Security**: API keys stored in environment variables
- **No Data Persistence**: Chat history not permanently stored

## 🚀 Performance Tips

- **PDF Optimization**: Use text-based PDFs for better results
- **File Size**: Smaller files process faster
- **Concurrent Users**: Application supports multiple simultaneous users
- **Memory Usage**: Large documents may require more RAM

## 🔄 Updates & Maintenance

### Updating Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Clearing Storage
```bash
# Remove vector stores
rmdir /s vector_stores

# Clear upload cache
rmdir /s uploads
mkdir uploads
```

## 📊 Monitoring & Logs

- **API Status**: Real-time monitoring via `/api/status`
- **Service Health**: Dashboard shows all service statuses
- **Error Handling**: Comprehensive error messages and logging
- **Performance**: Monitor response times in browser network tab

## 🤝 Contributing

StudyHub is designed to be extensible. To add new features:

1. Add routes in `app.py`
2. Create corresponding templates in `templates/`
3. Update navigation in `base.html`
4. Add styling in `static/css/style.css`
5. Add JavaScript in `static/js/main.js`

## 📝 License

This project is open source. Feel free to modify and distribute as needed.

## 🙏 Acknowledgments

- **LangChain**: Document processing framework
- **Ollama**: Local LLM hosting
- **Google Gemini**: AI summarization and analysis
- **Bootstrap**: UI framework
- **FAISS**: Vector similarity search

---

**Happy Learning with StudyHub! 🎓**

*For support, check the help section in the application or review the troubleshooting guide above.*