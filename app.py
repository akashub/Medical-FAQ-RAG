import streamlit as st
import os
from dotenv import load_dotenv
from rag_pipeline import MedicalRAGPipeline

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical FAQ Chatbot",
    page_icon="🏥",
    layout="wide"
)

def main():
    st.title("🏥 Medical FAQ Chatbot")
    st.markdown("*Powered by Google Gemini and TF-IDF Retrieval*")
    st.markdown("---")
    
    # Sidebar for API key input
    st.sidebar.title("Configuration")
    
    # Get API key from environment or user input
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key:
        gemini_api_key = st.sidebar.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
    
    if not gemini_api_key:
        st.warning("⚠️ Please enter your Gemini API key to start using the chatbot.")
        st.info("🔗 You can get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
        
        # Show setup instructions
        with st.expander("📋 Setup Instructions"):
            st.markdown("""
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Sign in with your Google account
            3. Click "Create API Key"
            4. Copy the API key and paste it in the sidebar
            5. Start chatting with the medical FAQ bot!
            """)
        return
    
    # Initialize the RAG pipeline
    @st.cache_resource
    def initialize_rag_pipeline(api_key):
        try:
            pipeline = MedicalRAGPipeline(api_key)
            pipeline.load_and_process_data("data/medical_faqs.json")
            return pipeline
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {str(e)}")
            return None
    
    try:
        rag_pipeline = initialize_rag_pipeline(gemini_api_key)
        
        if rag_pipeline is None:
            return
        
        # Main chat interface
        st.markdown("### 💬 Ask me any medical question!")
        st.markdown("*📝 Note: This chatbot provides general medical information. Always consult with healthcare professionals for personalized medical advice.*")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Accept user input
        if prompt := st.chat_input("Type your medical question here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Searching knowledge base and generating response..."):
                    response = rag_pipeline.chat(prompt)
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Sidebar with sample questions
        st.sidebar.markdown("### 💡 Sample Questions:")
        sample_questions = [
            "What are the early symptoms of diabetes?",
            "Can children take paracetamol?",
            "What foods are good for heart health?",
            "How much water should I drink daily?",
            "What are the symptoms of high blood pressure?",
            "How can I improve my sleep quality?",
            "What are the benefits of regular exercise?",
            "How do I know if I have a food allergy?"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.sidebar.button(f"❓ {question}", key=f"sample_{i}"):
                # Add the sample question as if the user typed it
                st.session_state.messages.append({"role": "user", "content": question})
                
                with st.spinner("🤔 Generating response..."):
                    response = rag_pipeline.chat(question)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        # Clear chat button
        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Show system info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ System Info")
        st.sidebar.markdown("**Retrieval:** BM25 + Dense Vectorization")
        st.sidebar.markdown("**Generation:** Google Gemini 2.5 Flash")
        st.sidebar.markdown("**Knowledge Base:** 43,000 Medical FAQs [https://www.kaggle.com/datasets/thedevastator/comprehensive-medical-q-a-dataset?resource=download]")
        
        # Footer
        st.markdown("---")
        st.markdown("**⚠️ Medical Disclaimer:** This chatbot provides general medical information only and should not replace professional medical advice. Always consult qualified healthcare professionals for medical diagnosis, treatment, or personalized advice.")
        
    except Exception as e:
        st.error(f"❌ Error initializing the chatbot: {str(e)}")
        st.info("Please check your API key and try again.")

if __name__ == "__main__":
    main()