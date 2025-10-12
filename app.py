from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import os
import tempfile
import logging
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import uuid

# Import functionality from individual apps
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# PDF Chat imports
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Summarizer imports
import requests
from bs4 import BeautifulSoup
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Quiz Generator imports
from markitdown import MarkItDown

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'studyhub_secret_key_2024_change_in_production')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 52428800))  # Default 50MB

# Ensure upload directory exists (use /tmp for Cloud Run)
if os.getenv('FLASK_ENV') == 'production':
    app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
    vector_store_base = '/tmp/vector_stores'
else:
    vector_store_base = 'vector_stores'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(vector_store_base, exist_ok=True)

# Global variables for caching models
embeddings = None
gemini_model = None

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'models/gemini-2.5-flash')

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        logger.info(f"✅ Gemini API configured successfully with model: {GEMINI_MODEL}")
    except Exception as e:
        logger.error(f"Error configuring Gemini API: {e}")
        gemini_model = None

def get_embeddings():
    """Load and cache the HuggingFace embeddings model with memory optimization."""
    global embeddings
    if embeddings is None:
        try:
            # Use a smaller, more memory-efficient model for Cloud Run
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': 1}
            )
            logger.info("✅ Embeddings model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {e}")
            # Fallback to basic embeddings if needed
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings



# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/pdf-chat')
def pdf_chat():
    """PDF Chat interface."""
    return render_template('pdf_chat.html')

@app.route('/summarizer')
def summarizer():
    """Content Summarizer interface."""
    return render_template('summarizer.html')

@app.route('/quiz-generator')
def quiz_generator():
    """Quiz Generator interface."""
    return render_template('quiz_generator.html')

# ============================================================================
# PDF CHAT FUNCTIONALITY
# ============================================================================

def extract_text_from_pdfs(pdf_files):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
    return text

def get_text_chunks(text):
    """Split text into chunks for processing with memory optimization."""
    # Use smaller chunks for better memory management in Cloud Run
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Reduced from 1000
        chunk_overlap=100  # Reduced from 200
    )
    chunks = text_splitter.split_text(text)
    logger.info(f"✅ Text split into {len(chunks)} chunks")
    return chunks

def create_vector_store(text_chunks, session_id):
    """Create FAISS vector store from text chunks with memory optimization."""
    try:
        embeddings_model = get_embeddings()
        
        # Process in smaller batches to reduce memory usage
        batch_size = 10
        all_vectors = []
        
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1}")
            
            if not all_vectors:
                # Create initial vector store with first batch
                vector_store = FAISS.from_texts(batch, embedding=embeddings_model)
            else:
                # Add subsequent batches to existing store
                batch_store = FAISS.from_texts(batch, embedding=embeddings_model)
                vector_store.merge_from(batch_store)
        
        # Save vector store for session (use /tmp in production)
        if os.getenv('FLASK_ENV') == 'production':
            vector_store_path = f"/tmp/vector_stores/session_{session_id}"
        else:
            vector_store_path = f"vector_stores/session_{session_id}"
        os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
        vector_store.save_local(vector_store_path)
        
        logger.info(f"✅ Vector store created successfully with {len(text_chunks)} chunks")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating vector store: {e}")
        return False

def get_vector_store(session_id):
    """Load vector store for session."""
    try:
        if os.getenv('FLASK_ENV') == 'production':
            vector_store_path = f"/tmp/vector_stores/session_{session_id}"
        else:
            vector_store_path = f"vector_stores/session_{session_id}"
        if os.path.exists(vector_store_path):
            embeddings_model = get_embeddings()
            vector_store = FAISS.load_local(
                vector_store_path,
                embeddings_model,
                allow_dangerous_deserialization=True
            )
            return vector_store.as_retriever()
        return None
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None

@app.route('/upload-pdfs', methods=['POST'])
def upload_pdfs():
    """Handle PDF file uploads and processing."""
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'})
        
        files = request.files.getlist('files')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'success': False, 'error': 'No files selected'})
        
        # Generate session ID if not exists
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Process PDFs
        pdf_files = []
        for file in files:
            if file and file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                pdf_files.append(filepath)
        
        if not pdf_files:
            return jsonify({'success': False, 'error': 'No valid PDF files found'})
        
        # Extract text
        with open(pdf_files[0], 'rb') as f:
            text = extract_text_from_pdfs([f])
        
        if not text.strip():
            return jsonify({'success': False, 'error': 'No text could be extracted from PDFs'})
        
        # Create chunks and vector store
        text_chunks = get_text_chunks(text)
        if create_vector_store(text_chunks, session_id):
            # Clean up uploaded files
            for filepath in pdf_files:
                try:
                    os.remove(filepath)
                except:
                    pass
            
            return jsonify({'success': True, 'message': 'PDFs processed successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to create vector store'})
            
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")
        return jsonify({'success': False, 'error': str(e)})

def get_relevant_context(retriever, question, max_docs=3):
    """Get relevant document chunks for the question."""
    try:
        docs = retriever.get_relevant_documents(question)
        # Limit to max_docs and combine the content
        relevant_docs = docs[:max_docs]
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        return context
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries using Gemini."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'success': False, 'error': 'No question provided'})
        
        if not gemini_model:
            return jsonify({'success': False, 'error': 'Gemini API not configured. Please check your API key.'})
        
        if 'session_id' not in session:
            return jsonify({'success': False, 'error': 'No documents uploaded. Please upload PDFs first.'})
        
        session_id = session['session_id']
        retriever = get_vector_store(session_id)
        
        if not retriever:
            return jsonify({'success': False, 'error': 'No documents found. Please upload PDFs first.'})
        
        # Get relevant context from the documents
        context = get_relevant_context(retriever, question)
        
        # Create prompt for Gemini
        prompt = f"""
        You are a helpful assistant that answers questions based on the provided document context. 
        Use the context to answer the question accurately. If the question cannot be answered from the context, 
        say so and provide a general response based on your knowledge.
        
        Context from documents:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        # Get response from Gemini
        response = gemini_model.generate_content(prompt)
        answer = response.text if response.text else 'Sorry, I could not generate an answer.'
        
        return jsonify({'success': True, 'answer': answer})
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return jsonify({'success': False, 'error': 'An error occurred while processing your question.'})

# ============================================================================
# SUMMARIZER FUNCTIONALITY
# ============================================================================

def extract_video_id(url):
    """Extract YouTube video ID from URL and convert to standard format."""
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([^&\n?#]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([^&\n?#]+)',
        r'(?:https?://)?(?:www\.)?youtu\.be/([^&\n?#\?]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            # Clean up any additional parameters
            if '&' in video_id:
                video_id = video_id.split('&')[0]
            if '?' in video_id:
                video_id = video_id.split('?')[0]
            return video_id
    return None

def get_youtube_page_info(video_url):
    """Get YouTube video information by scraping the video page."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(video_url, headers=headers, timeout=int(os.getenv('REQUEST_TIMEOUT', 10)))
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('meta', property='og:title')
        title = title_tag['content'] if title_tag else "Unknown Title"
        
        # Extract description - try multiple methods
        description = "No description available"
        
        # Method 1: og:description
        desc_tag = soup.find('meta', property='og:description')
        if desc_tag and desc_tag.get('content'):
            description = desc_tag['content']
        
        # Method 2: Twitter description
        if description == "No description available":
            twitter_desc = soup.find('meta', attrs={'name': 'twitter:description'})
            if twitter_desc and twitter_desc.get('content'):
                description = twitter_desc['content']
        
        # Method 3: Look for JSON-LD data
        if description == "No description available":
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, list):
                        data = data[0]
                    if 'description' in data:
                        description = data['description']
                        break
                except:
                    continue
        
        # Extract channel info
        channel_tag = soup.find('link', {'itemprop': 'name'})
        channel = channel_tag['content'] if channel_tag else "Unknown Channel"
        
        # If we still don't have good content, provide a helpful message
        if len(description) < 50 or description == "No description available":
            description = f"""This appears to be a YouTube video titled "{title}" from channel "{channel}". 
            However, I cannot access the full video content or transcript from this URL. 
            For better summarization, please provide:
            1. The full YouTube URL (not shortened)
            2. A transcript of the video
            3. Or paste the video's description/key points directly"""
        
        return {
            'title': title,
            'description': description,
            'channel': channel,
            'url': video_url
        }
    except Exception as e:
        logger.error(f"Error extracting YouTube info: {e}")
        return None

def get_webpage_content(url):
    """Extract text from webpage."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=int(os.getenv('REQUEST_TIMEOUT', 15)))
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]  # Limit to first 5000 characters
    except Exception as e:
        logger.error(f"Error extracting webpage content: {e}")
        return None

def process_with_gemini(content, content_type, summarize=True):
    """Process content with Gemini AI."""
    if not gemini_model:
        return "❌ Gemini API not configured. Please check your API key."
    
    try:
        if summarize:
            prompt = f"""
            Please provide a comprehensive summary of this {content_type}:
            
            {content}
            
            Include:
            1. Main topic/theme
            2. Key points (3-5 bullet points)
            3. Important details or insights
            4. Conclusion or takeaways
            
            Format your response clearly with headers and bullet points.
            """
        else:
            prompt = f"""
            Please provide a detailed analysis of this {content_type}:
            
            {content}
            
            Include a thorough breakdown of all important information, themes, and insights.
            """
        
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error with Gemini API: {e}")
        return f"❌ Error processing content: {str(e)}"

@app.route('/summarize', methods=['POST'])
def summarize_content():
    """Handle content summarization requests."""
    try:
        data = request.get_json()
        content_type = data.get('type', '')
        content_input = data.get('content', '').strip()
        summarize = data.get('summarize', True)
        
        if not content_input:
            return jsonify({'success': False, 'error': 'No content provided'})
        
        result = None
        
        if content_type == 'youtube':
            video_id = extract_video_id(content_input)
            if not video_id:
                return jsonify({'success': False, 'error': 'Invalid YouTube URL'})
            
            video_info = get_youtube_page_info(content_input)
            if video_info:
                content_text = f"Title: {video_info['title']}\nChannel: {video_info['channel']}\nDescription: {video_info['description']}"
                result = process_with_gemini(content_text, "YouTube video", summarize)
            else:
                return jsonify({'success': False, 'error': 'Could not extract video information'})
                
        elif content_type == 'webpage':
            content_text = get_webpage_content(content_input)
            if content_text:
                result = process_with_gemini(content_text, "webpage", summarize)
            else:
                return jsonify({'success': False, 'error': 'Could not extract webpage content'})
                
        elif content_type == 'text':
            result = process_with_gemini(content_input, "text content", summarize)
        
        if result:
            return jsonify({'success': True, 'result': result})
        else:
            return jsonify({'success': False, 'error': 'Failed to process content'})
            
    except Exception as e:
        logger.error(f"Error in summarize: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ============================================================================
# QUIZ GENERATOR FUNCTIONALITY
# ============================================================================

def extract_text_from_files_quiz(files):
    """Extract text from multiple file types for quiz generation."""
    combined_text = ""
    markitdown = MarkItDown()
    
    for file in files:
        try:
            if file.filename.endswith('.pdf'):
                # For PDF files
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    combined_text += page.extract_text() + "\n"
            else:
                # For other file types, use markitdown
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                    file.save(tmp_file.name)
                    result = markitdown.convert(tmp_file.name)
                    combined_text += result.text_content + "\n"
                    os.unlink(tmp_file.name)
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            continue
    
    return combined_text

def generate_questions_with_gemini(text, question_types, num_questions):
    """Generate questions using Gemini API."""
    if not gemini_model:
        return "❌ Gemini API not configured. Please check your API key.", ""
    
    try:
        # Prepare question type instructions
        type_instructions = []
        if "Multiple Choice" in question_types:
            type_instructions.append("- Multiple choice questions with 4 options (A, B, C, D) and indicate the correct answer")
        if "True/False" in question_types:
            type_instructions.append("- True/False questions with explanations")
        if "Essay" in question_types:
            type_instructions.append("- Essay questions that require detailed responses")
        
        prompt = f"""
        Based on the following text, create {num_questions} educational questions.
        
        Question types to include:
        {chr(10).join(type_instructions)}
        
        Text content:
        {text[:4000]}  # Limit text to avoid token limits
        
        Format your response as follows:
        
        QUESTIONS:
        1. [Question text]
        [If multiple choice, include options A, B, C, D]
        
        2. [Question text]
        [Continue numbering...]
        
        ANSWERS:
        1. [Answer with explanation]
        2. [Answer with explanation]
        [Continue numbering...]
        
        Make sure questions are educational, clear, and directly related to the content.
        """
        
        response = gemini_model.generate_content(prompt)
        result = response.text
        
        # Split questions and answers
        if "ANSWERS:" in result:
            questions_part, answers_part = result.split("ANSWERS:", 1)
            questions = questions_part.replace("QUESTIONS:", "").strip()
            answers = answers_part.strip()
        else:
            questions = result
            answers = "Answers not properly formatted."
        
        return questions, answers
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return f"❌ Error generating questions: {str(e)}", ""

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    """Handle quiz generation requests."""
    try:
        files = request.files.getlist('files')
        question_types = request.form.getlist('question_types')
        num_questions = int(request.form.get('num_questions', 5))
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'success': False, 'error': 'No files uploaded'})
        
        if not question_types:
            return jsonify({'success': False, 'error': 'No question types selected'})
        
        # Extract text from files
        text = extract_text_from_files_quiz(files)
        
        if not text.strip():
            return jsonify({'success': False, 'error': 'No text could be extracted from the uploaded files'})
        
        # Generate questions
        questions, answers = generate_questions_with_gemini(text, question_types, num_questions)
        
        return jsonify({
            'success': True,
            'questions': questions,
            'answers': answers
        })
        
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        return jsonify({'success': False, 'error': str(e)})

# ============================================================================
# API STATUS ENDPOINTS
# ============================================================================

@app.route('/api/status')
def api_status():
    """Return API status information."""
    status = {
        'gemini_api': gemini_model is not None,
        'embeddings_loaded': embeddings is not None,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    # Test Gemini API on startup
    if gemini_model:
        try:
            test_response = gemini_model.generate_content("Hello, this is a test.")
            logger.info("✅ Gemini API test successful!")
        except Exception as e:
            logger.error(f"❌ Gemini API test failed: {e}")
    
    logger.info("Starting StudyHub unified application...")
    
    # Get port from environment (for GCP) or default to 8080
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    
    app.run(debug=debug, host='0.0.0.0', port=port)