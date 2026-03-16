import streamlit as st
import requests
import json
from datetime import datetime
import os
from typing import List, Dict, Any
import pandas as pd
import base64
from PIL import Image
from io import BytesIO

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
ALLOWED_EXTENSIONS = ["pdf"]


class DellChatbotUI:
    """Streamlit UI for Dell Chatbot with image support"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.apply_custom_css()
    
    def setup_page_config(self):
        st.set_page_config(
            page_title="Dell Documentation Assistant",
            page_icon="💬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "session_id" not in st.session_state:
            st.session_state.session_id = None
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        if "api_healthy" not in st.session_state:
            st.session_state.api_healthy = True
    
    def apply_custom_css(self):
        """Apply ChatGPT-like styling with image support"""
        st.markdown("""
        <style>
        /* Main container */
        .main > div {
            padding: 0rem 1rem;
        }
        
        /* Chat message containers */
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        
        .user-message {
            background-color: #f0f2f6;
            border-left: 4px solid #0e5e6f;
        }
        
        .assistant-message {
            background-color: white;
            border-left: 4px solid #0a7e8c;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Message header */
        .message-header {
            display: flex;
            align-items: center;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .user-icon, .assistant-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 0.5rem;
        }
        
        .user-icon {
            background-color: #0e5e6f;
            color: white;
        }
        
        .assistant-icon {
            background-color: #0a7e8c;
            color: white;
        }
        
        /* Message content */
        .message-content {
            padding-left: 2.5rem;
            line-height: 1.6;
        }
        
        .message-content p {
            margin-bottom: 0.5rem;
        }
        
        /* Image container */
        .image-container {
            margin: 1rem 0;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 0.3rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .image-caption {
            margin-top: 0.5rem;
            font-size: 0.9rem;
            color: #666;
            font-style: italic;
        }
        
        /* Source citations */
        .source-citation {
            background-color: #e8f4f8;
            padding: 0.5rem;
            border-radius: 0.3rem;
            margin-top: 0.5rem;
            font-size: 0.9rem;
            border-left: 3px solid #0a7e8c;
        }
        
        .source-citation.has-image {
            border-left: 3px solid #28a745;
        }
        
        /* File upload area */
        .upload-area {
            border: 2px dashed #0a7e8c;
            border-radius: 0.5rem;
            padding: 2rem;
            text-align: center;
            background-color: #f8f9fa;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #0a7e8c;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.3rem;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #0e5e6f;
            color: white;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Status indicators */
        .status-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        
        .status-healthy {
            background-color: #d4edda;
            color: #155724;
        }
        
        .status-unhealthy {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        /* Tables */
        .dataframe {
            width: 100%;
            border-collapse: collapse;
        }
        
        .dataframe th {
            background-color: #0a7e8c;
            color: white;
            padding: 0.5rem;
            text-align: left;
        }
        
        .dataframe td {
            padding: 0.5rem;
            border-bottom: 1px solid #dee2e6;
        }
        
        .dataframe tr:hover {
            background-color: #f8f9fa;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def check_api_health(self):
        """Check if API is healthy"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                st.session_state.api_healthy = data.get("status") == "healthy"
            else:
                st.session_state.api_healthy = False
        except:
            st.session_state.api_healthy = False
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:
            st.image("https://i.imgur.com/7k12EPD.png", width=200)
            st.title("Dell Assistant")
            
            # API Status
            self.check_api_health()
            status_color = "status-healthy" if st.session_state.api_healthy else "status-unhealthy"
            status_text = "🟢 Connected" if st.session_state.api_healthy else "🔴 Disconnected"
            st.markdown(f'<span class="status-badge {status_color}">{status_text}</span>', 
                       unsafe_allow_html=True)
            
            st.divider()
            
            # Upload Section
            st.subheader("📤 Upload Documents")
            st.markdown("Upload Dell PDF documentation to get started")
            
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type=ALLOWED_EXTENSIONS,
                accept_multiple_files=True,
                key="pdf_uploader"
            )
            
            if uploaded_files:
                if st.button("📄 Process Files", use_container_width=True):
                    self.process_uploads(uploaded_files)
            
            st.divider()
            
            # Stats Section
            if st.session_state.uploaded_files:
                st.subheader("📊 Stats")
                total_texts = sum(f["texts"] for f in st.session_state.uploaded_files)
                total_tables = sum(f["tables"] for f in st.session_state.uploaded_files)
                total_images = sum(f["images"] for f in st.session_state.uploaded_files)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📝 Texts", total_texts)
                    st.metric("📊 Tables", total_tables)
                with col2:
                    st.metric("🖼️ Images", total_images)
                    st.metric("📚 Files", len(st.session_state.uploaded_files))
                
                # File list
                with st.expander("📋 Processed Files"):
                    for f in st.session_state.uploaded_files:
                        st.markdown(f"**{f['filename']}**")
                        st.caption(f"Texts: {f['texts']} | Tables: {f['tables']} | Images: {f['images']}")
            
            st.divider()
            
            # Controls
            st.subheader("⚙️ Controls")
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = None
                st.rerun()
            
            if st.button("🗑️ Clear All Data", use_container_width=True):
                if st.session_state.api_healthy:
                    try:
                        requests.post(f"{API_BASE_URL}/clear")
                        st.session_state.uploaded_files = []
                        st.session_state.messages = []
                        st.session_state.session_id = None
                        st.success("✅ All data cleared")
                        st.rerun()
                    except:
                        st.error("❌ Failed to clear data")
            
            st.divider()
            st.caption("© 2024 Dell Technologies")
    
    def process_uploads(self, uploaded_files):
        """Process uploaded files"""
        with st.spinner("📤 Uploading and processing files..."):
            files_to_send = []
            for file in uploaded_files:
                if file.size > MAX_FILE_SIZE:
                    st.error(f"❌ {file.name} exceeds 200MB limit")
                    continue
                files_to_send.append(("files", (file.name, file.getvalue(), "application/pdf")))
            
            if files_to_send:
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/ingest",
                        files=files_to_send
                    )
                    
                    if response.status_code == 201:  # Backend returns 201 for ingest
                        result = response.json()
                        # Backend returns aggregated results, create a summary entry
                        summary_result = {
                            "filename": f"{len(files_to_send)} files uploaded",
                            "texts": result.get("texts", 0),
                            "tables": result.get("tables", 0), 
                            "images": result.get("images", 0),
                            "total_chunks": result.get("indexed", 0)
                        }
                        if summary_result not in st.session_state.uploaded_files:
                            st.session_state.uploaded_files.append(summary_result)
                        st.success(f"✅ Successfully processed {result.get('pdfs_processed', 0)} files")
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
    
    def render_image(self, image_data: Dict[str, Any]):
        """Render an image from base64 data"""
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data["content"])
            image = Image.open(BytesIO(image_bytes))
            
            # Display image
            st.image(image, caption=f"From: {image_data['source_pdf']} (Page {image_data['page']})", 
                    use_container_width=True)
            
            # Add metadata
            st.caption(f"Size: {image_data['width']}x{image_data['height']} pixels")
            
        except Exception as e:
            st.error(f"Failed to display image: {e}")
    
    def render_chat_message(self, message: Dict[str, Any], index: int):
        """Render a single chat message with image support"""
        is_user = message["role"] == "user"
        icon = "👤" if is_user else "🤖"
        icon_class = "user-icon" if is_user else "assistant-icon"
        message_class = "user-message" if is_user else "assistant-message"
        
        with st.container():
            # Message header and content
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <div class="message-header">
                    <span class="{icon_class}">{icon}</span>
                    <span>{'You' if is_user else 'Assistant'}</span>
                </div>
                <div class="message-content">
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show images for assistant messages
            if not is_user and "images" in message and message["images"]:
                st.markdown("##### 🖼️ Images in Response")
                cols = st.columns(min(3, len(message["images"])))
                for idx, img_data in enumerate(message["images"]):
                    with cols[idx % 3]:
                        self.render_image(img_data)
            
            # Show sources for assistant messages
            if not is_user and "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for source in message["sources"]:
                        has_image_class = "has-image" if source.get("has_image") else ""
                        st.markdown(f"""
                        <div class="source-citation {has_image_class}">
                            <strong>📄 {source['source_pdf']}</strong><br>
                            📍 Page {source['page']} | Type: {source['type']}
                            {f"<br>📷 Contains image" if source.get("has_image") else ""}
                        </div>
                        """, unsafe_allow_html=True)
    
    def render_main_chat(self):
        """Render main chat interface with image support"""
        # Welcome message
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h1>🤖 Dell Documentation Assistant</h1>
                <p style="color: #666; font-size: 1.2rem;">
                    Upload Dell PDF documents and ask questions about them.<br>
                    The assistant can process and display images from your documents!
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Chat messages
        for i, message in enumerate(st.session_state.messages):
            self.render_chat_message(message, i)
        
        # Chat input
        if prompt := st.chat_input("Ask about Dell products..."):
            # Add user message
            user_message = {"role": "user", "content": prompt}
            st.session_state.messages.append(user_message)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/query",
                            json={
                                "question": prompt,
                                "k": 6,
                                "include_sources": True
                            }
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Add assistant message with images
                            assistant_message = {
                                "role": "assistant",
                                "content": data["answer"],
                                "sources": data["sources"],
                                "images": data["images"]
                            }
                            st.session_state.messages.append(assistant_message)
                            
                            # Rerun to show new message
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")
    
    def render_evaluation_tab(self):
        """Render evaluation interface"""
        st.header("📊 RAG Evaluation")
        
        if not st.session_state.uploaded_files:
            st.warning("⚠️ Please upload some documents first")
            return
        
        st.markdown("""
        Test the RAG system's performance using RAGAS metrics:
        - **Faithfulness**: How factual is the answer?
        - **Answer Relevancy**: How relevant is the answer?
        - **Context Precision**: How precise is the retrieved context?
        - **Context Recall**: How complete is the retrieved context?
        """)
        
        # Sample questions
        sample_questions = [
            "What are the specifications of Dell G16 7620?",
            "What are the setup instructions for Dell G16 7620?",
            "What ports are available on Dell G16 7620?",
            "Show me any images from the documentation"
        ]
        
        sample_answers = [
            "Dell G16 7620 specifications include regulatory model P105F and regulatory type P105F012/P105F011.",
            "The setup instructions are detailed in the manual covering initial setup and configuration.",
            "The port configuration is detailed in the specifications section of the manual.",
            "Images from the documentation are displayed when available."
        ]
        
        st.subheader("📝 Sample Test Cases")
        
        questions = []
        ground_truth = []
        
        for i, (q, a) in enumerate(zip(sample_questions, sample_answers)):
            col1, col2 = st.columns(2)
            with col1:
                questions.append(st.text_input(f"Question {i+1}", value=q, key=f"q_{i}"))
            with col2:
                ground_truth.append(st.text_input(f"Expected Answer {i+1}", value=a, key=f"a_{i}"))
        
        if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running evaluation... This may take a minute"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/evaluate",
                        json={
                            "questions": questions,
                            "ground_truth": ground_truth
                        }
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Faithfulness", f"{results['faithfulness']:.3f}")
                        with col2:
                            st.metric("Answer Relevancy", f"{results['answer_relevancy']:.3f}")
                        with col3:
                            st.metric("Context Precision", f"{results['context_precision']:.3f}")
                        with col4:
                            st.metric("Context Recall", f"{results['context_recall']:.3f}")
                        
                        # Individual scores
                        st.subheader("📋 Individual Scores")
                        if results.get("individual_scores"):
                            df = pd.DataFrame(results["individual_scores"])
                            st.dataframe(df, use_container_width=True)
                    else:
                        st.error(f"❌ Evaluation failed: {response.text}")
                except Exception as e:
                    st.error(f"❌ Evaluation error: {str(e)}")
    
    def run(self):
        """Main entry point"""
        self.render_sidebar()
        
        # Main content
        tabs = st.tabs(["💬 Chat", "📊 Evaluate"])
        
        with tabs[0]:
            self.render_main_chat()
        
        with tabs[1]:
            self.render_evaluation_tab()


if __name__ == "__main__":
    app = DellChatbotUI()
    app.run()