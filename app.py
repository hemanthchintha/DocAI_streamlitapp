import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from io import BytesIO
import PyPDF2

# Configure page
st.set_page_config(
    page_title="Doc AI - Pediatric Assistant",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display visible title
st.title("Doc AI - Pediatric Assistant 👶")

# Custom CSS
st.markdown("""
<style>
    .user-message {
        text-align: right;
        margin-left: 20%;
        margin-right: 0;
        padding: 10px;
        border-radius: 10px 10px 0 10px;
        background-color: #e3f2fd;
    }
    .assistant-message {
        text-align: left;
        margin-right: 20%;
        margin-left: 0;
        padding: 10px;
        border-radius: 10px 10px 10px 0;
        background-color: #f5f5f5;
    }
    div[data-testid="stChatInputContainer"] {
        width: 400px !important;
        max-width: 400px !important;
    }
    
    /* Style the actual input box */
    div[data-testid="stChatInputContainer"] textarea {
        width: 100% !important;
        min-width: 100% !important;
    }
    
    /* Position the container (remove fixed positioning) */
    div[data-testid="stChatInputContainer"] {
        position: relative !important;
        left: 0 !important;
        margin-left: 0 !important;
        padding-left: 0 !important;
    }
    .pdf-display {
        padding: 10px;
        background-color: #fff3e0;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .new-chat-btn {
        margin-top: 10px;
        width: 100%;
    }
    /* Adjust for mobile view */
    @media (max-width: 768px) {
        .stChatInput {
            width: 90% !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
@st.cache_resource
def init_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("API key not found. Please set OPENAI_API_KEY in your .env file.")
        st.stop()
    return OpenAI(api_key=api_key)

client = init_client()

# Document processing function
def process_document(uploaded_file):
    """Extract text from PDF using PyPDF2"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        extracted_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        return extracted_text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # New Chat button - fixed version
    if st.button("🔄 New Chat", 
                key="new_chat", 
                help="Start a new conversation",
                type="primary"):
        st.session_state.messages = []
        st.session_state.document_ready = False
        st.session_state.document_text = None
        st.rerun()
    
    st.markdown("---")
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload medical documents (PDF only)",
        type=["pdf"],
        accept_multiple_files=False
    )
    
    SYSTEM_PROMPT ="""You are a pediatric care assistant. Provide:
- Clear, accurate pediatric advice 
- Simple language for parents
- Markdown formatting for readability
- Warnings for serious symptoms
- Always recommend professional care when needed
- use minimum words to answer the question"""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

# Process uploaded document
if uploaded_file and (st.session_state.get("last_uploaded") != uploaded_file.name):
    with st.spinner("Reading document..."):
        extracted_text = process_document(uploaded_file)
        if extracted_text:
            st.session_state.document_text = extracted_text
            st.session_state.last_uploaded = uploaded_file.name
            st.session_state.document_ready = True
            st.sidebar.success("Document ready for reference!")
            st.sidebar.markdown(f'<div class="pdf-display">📄 {uploaded_file.name}</div>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about your child's health..."):
    # Only include document if user asks about it
    include_document = any(keyword in prompt.lower() for keyword in ["document", "report", "file", "pdf", "attachment"])
    
    # Build prompt
    full_prompt = prompt
    if include_document and st.session_state.document_ready:
        full_prompt = f"DOCUMENT CONTEXT:\n{st.session_state.document_text}\n\nUSER QUESTION: {prompt}"
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)
    
    # Generate and display assistant response
    with st.spinner("Analyzing..."):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend([{"role": m["role"], "content": m["content"]} 
                      for m in st.session_state.messages])
        
        # For document questions, replace last user message with the full prompt
        if include_document and st.session_state.document_ready:
            messages[-1]["content"] = full_prompt
        
        response = st.write_stream(
            client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.7,
                stream=True
            )
        )
    
    st.session_state.messages.append({"role": "assistant", "content": response})
