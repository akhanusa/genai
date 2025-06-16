import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# Hardcoded credentials for simplicity
AUTHORIZED_USERS = {
    "admin": "password123",  # Replace with secure password
    # Add more users if needed
}

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Authentication logic
if not st.session_state.authenticated:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username] == password:
            st.session_state.authenticated = True
            st.success("Login successful.")
        else:
            st.error("Invalid credentials")
    st.stop()  # Prevent rest of app from running
else:
    st.title("Codebasics Q&A 🌱")

    # Only show button to authorized users
    btn = st.button("Create Knowledgebase")
    if btn:
        create_vector_db()

    question = st.text_input("Question: ")
    if question:
        chain = get_qa_chain()
        response = chain(question)
        st.header("Answer")
        st.write(response["result"])