import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# --- Mock credential store ---
AUTHORIZED_USERS = {
    "admin": "password123",
    # Add more users if needed
}

# --- Session State Initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "question" not in st.session_state:
    st.session_state.question = ""

# --- Title ---
st.title("Codebasics Q&A 🌱")

# --- Logout Option ---
if st.session_state.authenticated:
    st.write(f"Logged in as: **{st.session_state.username}**")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.experimental_rerun()

# --- Login Section ---
with st.expander("🔐 Login"):
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        if username in AUTHORIZED_USERS and AUTHORIZED_USERS[username] == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

# --- Main Q&A Interface ---
st.text_input("Ask a question:", key="question_input", value=st.session_state.question)

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Submit"):
        question = st.session_state.question_input
        if question.strip():
            st.session_state.question = question
            chain = get_qa_chain()
            response = chain(question)
            st.header("Answer")
            st.write(response["result"])
with col2:
    if st.button("Clear"):
        st.session_state.question = ""
        st.session_state.question_input = ""
        st.experimental_rerun()

# --- Show Knowledgebase Button if Admin Logged In ---
if st.session_state.authenticated and st.session_state.username == "admin":
    st.markdown("---")
    if st.button("📚 Create Knowledgebase"):
        create_vector_db()
        st.success("Knowledgebase created/updated!")