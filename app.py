from __future__ import annotations

from rag_pipeline import langsmith_rag
from typing import Dict, Optional

import time
import uuid
import logging
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI-Powered မြန်မာအဏုဇီဝဗေဒ ကျွမ်းကျင်သူ 🚀",
    page_icon="💬",
    layout="wide"
)

# Constants
MAX_CONVERSATIONS = 10
MAX_MESSAGES_PER_CONVERSATION = 100
MAX_MESSAGE_LENGTH = 5000
CONVERSATION_TIMEOUT_HOURS = 24

class ConversationManager:
    """Manages conversation state and operations"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize session state variables with validation"""
        try:
            if "conversations" not in st.session_state:
                st.session_state.conversations = {}
            
            if "current_conversation_id" not in st.session_state:
                ConversationManager.create_first_conversation()
                
            # Clean up old conversations
            ConversationManager.cleanup_old_conversations()
            
        except Exception as e:
            logger.error(f"Error initializing session state: {str(e)}")
            st.error("Failed to initialize application. Please refresh the page.")
    
    @staticmethod
    def create_first_conversation():
        """Create the initial conversation"""
        conversation_id = str(uuid.uuid4())
        st.session_state.conversations[conversation_id] = {
            "id": conversation_id,
            "name": "New Chat",
            "messages": [],
            "created_at": time.time(),
            "updated_at": time.time()
        }
        st.session_state.current_conversation_id = conversation_id
        logger.info(f"Created first conversation: {conversation_id}")
    
    @staticmethod
    def create_new_conversation() -> str:
        """Create a new conversation with validation"""
        try:
            # Check conversation limit
            if len(st.session_state.conversations) >= MAX_CONVERSATIONS:
                st.warning(f"Maximum {MAX_CONVERSATIONS} conversations allowed. Please delete some conversations first.")
                return None
            
            conversation_id = str(uuid.uuid4())
            st.session_state.conversations[conversation_id] = {
                "id": conversation_id,
                "name": "New Chat",
                "messages": [],
                "created_at": time.time(),
                "updated_at": time.time()
            }
            st.session_state.current_conversation_id = conversation_id
            
            logger.info(f"Created new conversation: {conversation_id}")
            st.rerun()
            return conversation_id
            
        except Exception as e:
            logger.error(f"Error creating new conversation: {str(e)}")
            st.error("Failed to create new conversation. Please try again.")
            return None
    
    @staticmethod
    def delete_conversation(conversation_id: str) -> bool:
        """Delete a conversation with validation"""
        try:
            if len(st.session_state.conversations) <= 1:
                st.warning("Cannot delete the last conversation.")
                return False
            
            if conversation_id not in st.session_state.conversations:
                st.warning("Conversation not found.")
                return False
            
            del st.session_state.conversations[conversation_id]
            
            # Switch to another conversation if current one was deleted
            if st.session_state.current_conversation_id == conversation_id:
                available_conversations = list(st.session_state.conversations.keys())
                if available_conversations:
                    st.session_state.current_conversation_id = available_conversations[0]
                else:
                    ConversationManager.create_first_conversation()
            
            logger.info(f"Deleted conversation: {conversation_id}")
            st.rerun()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            st.error("Failed to delete conversation. Please try again.")
            return False
    
    @staticmethod
    def switch_conversation(conversation_id: str) -> bool:
        """Switch to a different conversation with validation"""
        try:
            if conversation_id not in st.session_state.conversations:
                st.error("Conversation not found.")
                return False
            
            st.session_state.current_conversation_id = conversation_id
            logger.info(f"Switched to conversation: {conversation_id}")
            st.rerun()
            return True
            
        except Exception as e:
            logger.error(f"Error switching to conversation {conversation_id}: {str(e)}")
            st.error("Failed to switch conversation. Please try again.")
            return False
    
    @staticmethod
    def add_message_to_current_conversation(role: str, content: str) -> bool:
        """Add message to current conversation with validation"""
        try:
            if not content or not content.strip():
                return False
            
            if len(content) > MAX_MESSAGE_LENGTH:
                st.warning(f"Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed.")
                return False
            
            current_conv_id = st.session_state.current_conversation_id
            current_conv = st.session_state.conversations[current_conv_id]
            
            # Check message limit per conversation
            if len(current_conv["messages"]) >= MAX_MESSAGES_PER_CONVERSATION:
                st.warning(f"Maximum {MAX_MESSAGES_PER_CONVERSATION} messages per conversation reached.")
                return False
            
            current_conv["messages"].append({
                "role": role,
                "content": content.strip(),
                "timestamp": time.time(),
                "id": str(uuid.uuid4())
            })
            current_conv["updated_at"] = time.time()
            
            # Update conversation name based on first user message
            if (role == "user" and 
                current_conv["name"] == "New Chat" and 
                len([m for m in current_conv["messages"] if m["role"] == "user"]) == 1):
                current_conv["name"] = content[:30] + ("..." if len(content) > 30 else "")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}")
            st.error("Failed to add message. Please try again.")
            return False
    
    @staticmethod
    def get_current_conversation() -> Optional[Dict]:
        """Get current conversation data with validation"""
        try:
            current_id = st.session_state.current_conversation_id
            if current_id in st.session_state.conversations:
                return st.session_state.conversations[current_id]
            else:
                logger.warning(f"Current conversation {current_id} not found")
                ConversationManager.create_first_conversation()
                return st.session_state.conversations[st.session_state.current_conversation_id]
        except Exception as e:
            logger.error(f"Error getting current conversation: {str(e)}")
            return None
    
    @staticmethod
    def cleanup_old_conversations():
        """Remove conversations older than timeout period"""
        try:
            current_time = time.time()
            timeout_seconds = CONVERSATION_TIMEOUT_HOURS * 3600
            
            conversations_to_remove = []
            for conv_id, conv_data in st.session_state.conversations.items():
                if current_time - conv_data.get("updated_at", 0) > timeout_seconds:
                    conversations_to_remove.append(conv_id)
            
            for conv_id in conversations_to_remove:
                if len(st.session_state.conversations) > 1:  # Keep at least one
                    del st.session_state.conversations[conv_id]
                    logger.info(f"Cleaned up old conversation: {conv_id}")
            
            # Ensure current conversation still exists
            if st.session_state.current_conversation_id not in st.session_state.conversations:
                if st.session_state.conversations:
                    st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
                else:
                    ConversationManager.create_first_conversation()
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            

def render_sidebar():
    """Render the conversation management sidebar"""
    with st.sidebar:
        st.title("💬 Chats")
        
        # New chat button
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            ConversationManager.create_new_conversation()
        
        st.markdown("---")
        
        try:
            # List all conversations
            conversations = sorted(
                st.session_state.conversations.values(),
                key=lambda x: x["updated_at"],
                reverse=True
            )
            
            for conv in conversations:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    is_active = conv["id"] == st.session_state.current_conversation_id
                    button_type = "primary" if is_active else "secondary"
                    
                    if st.button(
                        conv["name"],
                        key=f"conv_{conv['id']}",
                        use_container_width=True,
                        type=button_type
                    ):
                        ConversationManager.switch_conversation(conv["id"])
                
                with col2:
                    # Only show delete button if more than one conversation exists
                    if len(st.session_state.conversations) > 1:
                        if st.button(
                            "🗑️",
                            key=f"delete_{conv['id']}",
                            help="Delete conversation",
                            use_container_width=True
                        ):
                            ConversationManager.delete_conversation(conv["id"])
                    
        except Exception as e:
            logger.error(f"Error rendering conversations: {str(e)}")
            st.error("Error loading conversations")
        
def get_suggested_questions():
    """Get top 3 suggested questions in both English and Burmese"""
    return [
        {
            "english": "What is microbiology? Please explain.",
            "burmese": "အဏုဇီဝဗေဒ ဆိုတာ ဘာလဲ ရှင်းပြပါ",
            "display": "🔬 What is microbiology? / အဏုဇီဝဗေဒ ဆိုတာ ဘာလဲ ရှင်းပြပါ"
        },
        {
            "english": "Describe the structure of bacteria",
            "burmese": "Bacteria ၏ ဖွဲ့စည်းပုံကို ဖော်ပြပါ",
            "display": "🦠 Describe the structure of bacteria / Bacteria ၏ ဖွဲ့စည်းပုံကို ဖော်ပြပါ"
        },
        {
            "english": "What are the main components of viruses?",
            "burmese": "Virus ၏ အဓိက အစိတ်အပိုင်းများကို ဖော်ပြပါ",
            "display": "🦠 What are the main components of viruses? / Virus ၏ အဓိက အစိတ်အပိုင်းများကို ဖော်ပြပါ"
        }
    ]

def handle_suggested_question_click(question_data):
    """Handle when a suggested question is clicked"""
    # Use Burmese version for processing (assuming your RAG system handles both)
    question_to_process = question_data["burmese"]
    
    # Add user message
    if ConversationManager.add_message_to_current_conversation("user", question_to_process):
        # Generate response
        try:
            response = langsmith_rag(question_to_process)
            if response:
                ConversationManager.add_message_to_current_conversation("assistant", response)
                logger.info(f"Successfully generated response for suggested question: {question_to_process}")
            else:
                error_msg = "စိတ်မကောင်းပါဘူး၊ အဖြေမပေးနိုင်ပါ။ ထပ်မံကြိုးစားကြည့်ပါ။"
                ConversationManager.add_message_to_current_conversation("assistant", error_msg)
        except Exception as e:
            error_msg = f"စိတ်မကောင်းပါဘူး၊ အမှားတစ်ခုကြုံတွေ့ရပါတယ်: {str(e)}"
            logger.error(f"RAG function error for suggested question: {str(e)}")
            ConversationManager.add_message_to_current_conversation("assistant", error_msg)
        
        st.rerun()
  
def render_suggested_questions(current_conv):
    """Render suggested questions with dark styled design"""
    message_count = len(current_conv["messages"])
    suggested_questions = get_suggested_questions()

    if message_count == 0:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #232526, #414345);
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    margin-bottom: 1.5rem;
                    border-radius: 12px;
                    color: white;">
            <h3 style="text-align:center; margin-bottom: 1rem; color:#f5f5f5;">
                💭 Try asking these questions / ဤမေးခွန်းများကို စမ်းမေးကြည့်ပါ
            </h3>
        </div>
        """, unsafe_allow_html=True)

        # Render buttons in styled container
        for i, q in enumerate(suggested_questions):
            if st.button(q["display"], key=f"suggested_q_{i}", use_container_width=True):
                handle_suggested_question_click(q)

        st.markdown("---")
    
def render_app_info():
    """Render app information section"""
    st.html("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 12px; 
                margin-bottom: 2rem;
                color: white;
                font-family: 'Segoe UI', sans-serif;">
        
        <h2 style="color: white; text-align: center; margin-bottom: 1rem;">
            🤖 AI-Powered မြန်မာအဏုဇီဝဗေဒ ကျွမ်းကျင်သူ 🚀
        </h2>
        
        <p style="font-size: 1.1rem; text-align: center; margin-bottom: 1.5rem; line-height: 1.6; color: #f5f5f5;">
            အဆင့်မြင့် AI နည်းပညာကို အသုံးပြု၍ တည်ဆောက်ထားသော မြန်မာအဏုဇီဝဗေဒ လမ်းညွှန်စနစ်။ 
            ရှုပ်ထွေးသော အဏုဇီဝဗေဒ သဘောတရားများကို ရိုးရှင်းစွာ ရှင်းပြပေးနိုင်သည့် ထူးခြားသော ဒီဇိုင်း။
        </p>
        
        <div style="background: rgba(255,255,255,0.12); 
                    padding: 1.5rem; 
                    border-radius: 10px; 
                    backdrop-filter: blur(8px);">
            
            <h3 style="color: white; margin-bottom: 1rem; text-align: center;">
                ⚡ နောက်ဆုံးပေါ် နည်းပညာများ:
            </h3>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-top: 1rem;">
                
                <div style="background: rgba(255,255,255,0.15); 
                           padding: 1rem; 
                           border-radius: 8px;
                           text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔄</div>
                    <strong>RAG စနစ်</strong><br>
                    <small style="color:#e0e0e0;">Retrieval-Augmented Generation</small>
                </div>
                
                <div style="background: rgba(255,255,255,0.15); 
                           padding: 1rem; 
                           border-radius: 8px;
                           text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🇲🇲</div>
                    <strong>မြန်မာဘာသာ</strong><br>
                    <small style="color:#e0e0e0;">သဘာဝ ဘာသာပြန်ဆိုမှု</small>
                </div>
                
                <div style="background: rgba(255,255,255,0.15); 
                           padding: 1rem; 
                           border-radius: 8px;
                           text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⚡</div>
                    <strong>လျင်မြန်သော</strong><br>
                    <small style="color:#e0e0e0;">ရလဒ်များ</small>
                </div>
                
                <div style="background: rgba(255,255,255,0.15); 
                           padding: 1rem; 
                           border-radius: 8px;
                           text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📚</div>
                    <strong>အဆက်မပြတ်</strong><br>
                    <small style="color:#e0e0e0;">လေ့လာမှု အတွေ့အကြုံ</small>
                </div>
                
            </div>
        </div>
    </div>
    """)

def render_app_info_simple():
    """Render simple app information section"""
    st.info("""
    🤖  **AI-Powered မြန်မာအဏုဇီဝဗေဒ ကျွမ်းကျင်သူ** 🚀
    
    အဆင့်မြင့် AI နည်းပညာကို အသုံးပြု၍ တည်ဆောက်ထားသော မြန်မာအဏုဇီဝဗေဒ လမ်းညွှန်စနစ်။ 
    ရှုပ်ထွေးသော အဏုဇီဝဗေဒ သဘောတရားများကို ရိုးရှင်းစွာ ရှင်းပြပေးနိုင်သည့် ထူးခြားသော ဒီဇိုင်း။
    
    **⚡ နောက်ဆုံးပေါ် နည်းပညာများ:**
    - 🔄 RAG (Retrieval-Augmented Generation) စနစ်
    - 🇲🇲 မြန်မာဘာသာ သဘာဝ ဘာသာပြန်ဆိုမှု  
    - ⚡ လျင်မြန်သော ရလဒ်များ
    - 📚 အဆက်မပြတ် လေ့လာမှု အတွေ့အကြုံ
    """)

def render_chat_interface():
    """Render the main chat interface with suggested questions"""
    current_conv = ConversationManager.get_current_conversation()
    
    if not current_conv:
        st.error("Failed to load conversation. Please refresh the page.")
        return
    
    # Show app info only for new conversations (0 messages)
    if len(current_conv["messages"]) == 0:
        render_app_info()
        
    # Show suggested questions
    render_suggested_questions(current_conv)
    
    # Display chat messages in a container for better performance
    chat_container = st.container()
    
    with chat_container:
        try:
            messages = current_conv.get("messages", [])
            for i, message in enumerate(messages):
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                        
        except Exception as e:
            logger.error(f"Error displaying messages: {str(e)}")
            st.error("Error loading messages. Please refresh the page.")
    
    # Chat input with validation
    if prompt := st.chat_input("Type your question here... / သင်၏မေးခွန်းကို ဤနေရာတွင်ရိုက်ပါ...", max_chars=MAX_MESSAGE_LENGTH):
        try:
            # Validate input
            if not prompt.strip():
                st.warning("Please enter a valid message. / မှန်ကန်သောစာတစ်ကြောင်းရေးပါ။")
                return
            
            # Add user message
            if ConversationManager.add_message_to_current_conversation("user", prompt):
                
                # Display user message immediately
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking... / စဉ်းစားနေသည်..."):
                        try:
                            # Call the RAG function with error handling
                            response = langsmith_rag(prompt)
                            
                            if response:
                                st.write(response)
                                ConversationManager.add_message_to_current_conversation("assistant", response)
                                logger.info("Successfully generated response")
                            else:
                                error_msg = "စိတ်မကောင်းပါဘူး၊ အဖြေမပေးနိုင်ပါ။ ထပ်မံကြိုးစားကြည့်ပါ။"
                                st.error(error_msg)
                                ConversationManager.add_message_to_current_conversation("assistant", error_msg)
                                
                        except Exception as e:
                            error_msg = f"စိတ်မကောင်းပါဘူး၊ အမှားတစ်ခုကြုံတွေ့ရပါတယ်: {str(e)}"
                            logger.error(f"RAG function error: {str(e)}")
                            st.error(error_msg)
                            ConversationManager.add_message_to_current_conversation("assistant", error_msg)
                
                st.rerun()
            
        except Exception as e:
            logger.error(f"Error processing chat input: {str(e)}")
            st.error("Failed to process your message. Please try again. / သင်၏စာကိုမလုပ်ဆောင်နိုင်ပါ။ ထပ်မံကြိုးစားပါ။")

def main():
    """Main application function"""
    ConversationManager.initialize_session_state()
    
    # Render sidebar and main interface
    render_sidebar()
    render_chat_interface()

if __name__ == "__main__":
    main()