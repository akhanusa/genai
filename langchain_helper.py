from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import os
import streamlit as st

# Read Gemini API key from Streamlit secrets
gemini_api_key = st.secrets["GEMINI_API_KEY"]

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_api_key
)

# Embedding model configuration
model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

instructor_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Vector DB directory for Chroma
vectordb_dir = "chroma_db"

# Function to create Chroma vector store with chunking
def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faq.csv', source_column="prompt")
    data = loader.load()

    # Text chunking for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(data)

    # Create or overwrite Chroma vector database
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=instructor_embeddings,
        persist_directory=vectordb_dir
    )
    vectordb.persist()  # Ensures persistence to disk

# Function to load Chroma DB and return a RetrievalQA chain
def get_qa_chain():
    vectordb = Chroma(
        embedding_function=instructor_embeddings,
        persist_directory=vectordb_dir
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4, "score_threshold": 0.7}
    )

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from the "response" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

# Optional: build once, skip on next run if DB exists
if __name__ == "__main__":
    if not os.path.exists(vectordb_dir):
        create_vector_db()
    chain = get_qa_chain()
    response = chain("Do you have javascript course?")
    print(response["result"])
