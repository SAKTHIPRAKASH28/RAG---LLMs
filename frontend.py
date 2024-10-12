import streamlit as st
import requests

# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"  # Change this to your FastAPI server URL

def upload_file():
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "pptx", "txt", "docx", "png", "jpg", "jpeg", "mp3", "wav"])
    
    if uploaded_file is not None:
        if st.sidebar.button("Process File"):
            with st.spinner("Processing file..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                response = requests.post(f"{BACKEND_URL}/upload-file/", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.sidebar.success("File processed successfully!")
                    st.session_state.session_id = result["session_id"]
                    st.session_state.file_processed = True
                else:
                    st.sidebar.error(f"Error: {response.text}")

def close_session():
    if st.sidebar.button("Close Session"):
        if "session_id" in st.session_state:
            with st.spinner("Closing session..."):
                data = {"session_id": st.session_state.session_id}
                response = requests.post(f"{BACKEND_URL}/close-session/", data=data)
                
                if response.status_code == 200:
                    st.sidebar.success("Session closed successfully!")
                    del st.session_state.session_id
                    st.session_state.file_processed = False
                    st.session_state.messages = []
                else:
                    st.sidebar.error(f"Error: {response.text}")
        else:
            st.sidebar.warning("No active session to close.")

def handle_user_input():
    question = st.session_state.user_input
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        
        if "session_id" in st.session_state:
            with st.spinner("Generating response..."):
                data = {
                    "session_id": st.session_state.session_id,
                    "question": question,
                    "models": st.session_state.selected_models
                }
                response = requests.post(f"{BACKEND_URL}/ask-question/", data=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Store AI responses in session state for display
                    st.session_state.ai_responses = result["responses"]
                    
                    # Append the assistant response to the chat
                    response_text = "\n\n".join(
                        [f"<span style='color: #1aa3ff; font-weight: bold;'>{model.upper()}</span>: {response}"
                         for model, response in result["responses"].items()])
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    st.error(f"Error: {response.text}")
        
        # Clear the input box
        st.session_state.user_input = ""

# Function to display responses using the new design
import pyperclip

def display_responses():
    if "ai_responses" in st.session_state:
        st.subheader("AI Model Responses:")
        
        show_comparison = st.checkbox("Show side-by-side comparison")
        
        if show_comparison:
            cols = st.columns(len(st.session_state.ai_responses))
            for idx, (model, response_text) in enumerate(st.session_state.ai_responses.items()):
                with cols[idx]:
                    st.markdown(
                        f'<span style="color: #1aa3ff; font-weight: bold; font-size: 20px;">{model.upper()}</span>',
                        unsafe_allow_html=True
                    )
                    st.write(response_text)
                    
                    # Add a copy button for the response
                    if st.button("Copy", key=f"copy-{model}"):
                        pyperclip.copy(response_text)
                        st.success("Response copied to clipboard!")

        else:
            for model, response_text in st.session_state.ai_responses.items():
                with st.expander(f"{model}"):
                    st.markdown(
                        f'<span style="color: #1aa3ff; font-weight: bold; font-size: 20px;">{model.upper()}</span>',
                        unsafe_allow_html=True
                    )
                    st.write(response_text)
                    
                    # Add a copy button for the response
                    if st.button("Copy", key=f"copy-{model}"):
                        pyperclip.copy(response_text)
                        st.success("Response copied to clipboard!")


def main():
    st.set_page_config(page_title="Document Analysis & Q&A System", page_icon="📄")
    st.title("Retrieval-Augmented Generation (RAG) Document Analysis and Question-Answering System")
    st.markdown(
        """
        Welcome to the RAG Document Analysis and Q&A System! 
        This platform leverages advanced Retrieval-Augmented Generation technology to allow users to upload documents, interact with a smart AI model, and receive precise, context-driven answers.
       
        """
    )
    
    if "file_processed" not in st.session_state:
        st.session_state.file_processed = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    upload_file()
    
    if st.session_state.file_processed:
        close_session()

        # Select available models for generating responses
        st.session_state.selected_models = st.sidebar.multiselect("Select models", ["phi-3-small", "mistral", "meta-llama-3", "gpt-4o", "gemini", "ai21-jamba-1.5-mini"], default=["gpt-4o"])

        # Display user input and AI responses
        if st.session_state.messages:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div style="background-color: #1c1c1e; padding: 10px; border-radius: 10px; margin: 5px 0; color: white;">👤 {message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="background-color: #2c2c2e; padding: 10px; border-radius: 10px; margin: 5px 0; color: white;">🤖 {message["content"]}</div>', unsafe_allow_html=True)

        # Create a container for the input field and button
        input_container = st.container()
        with input_container:
            col1, col2 = st.columns([6, 1])
            with col1:
                st.text_input("", key="user_input", placeholder="Ask a question...",label_visibility="hidden")
            with col2:
                st.button("Send", on_click=handle_user_input)

        # Display AI responses
    

    else:
        st.markdown(
            """
            <p style="color: #FF6347; font-size: 16px;">
            Please upload and process a file to start the conversation.
            </p>
            """, 
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
