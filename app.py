import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --- Configuration & Initialization ---
DB_DIR = "./chroma_db"

# Initialize embedding model once (cached to prevent reloading on every rerun)
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# Initialize LLM
@st.cache_resource
def load_llm():
    return ChatMistralAI(model='mistral-small-2603', temperature=0.7)

llm = load_llm()

# Helper function to completely clear the old Vector DB safely on Windows
def clear_vector_store():
    # If the vectorstore is active in the session state, delete/close it first
    if "vectorstore" in st.session_state and st.session_state["vectorstore"] is not None:
        try:
            # LangChain Chroma wrapper exposes ._client which has a close method
            st.session_state["vectorstore"]._client.close()
        except Exception:
            pass
        st.session_state["vectorstore"] = None
        
    # Extra safety garbage collection to release file hooks
    import gc
    gc.collect()

    # Now attempt deletion
    if os.path.exists(DB_DIR):
        try:
            shutil.rmtree(DB_DIR)
        except PermissionError:
            # Fallback if Windows is still being stubborn: clear individual files
            # or inform the user to click the button once more to retry
            st.warning("Database was locked. Resetting... Please click 'Process' again to complete upload.")
            return False
    return True

# --- Streamlit UI Layout ---
st.set_page_config(page_title="Book RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Chat with Your Book")
st.subheader("Upload a PDF and ask questions using Mistral AI & LangChain")

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload your book (PDF)", type="pdf")
    
    process_button = st.button("Process & Build Knowledge Base", type="primary")

    if process_button and uploaded_file is not None:
        with st.spinner("Processing PDF... This might take a moment."):
            # 1. Save uploaded file to a temporary location
            temp_pdf_path = f"./temp_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 2. Clear old vector store safely without Windows crashing
            db_cleared = clear_vector_store()
            
            if db_cleared:
                # 3. Load and split the document
                loader = PyMuPDFLoader(temp_pdf_path)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                
                # 4. Create and persist Chroma Vectorstore
                vectorstore = Chroma.from_documents(
                    documents=docs, 
                    embedding=embedding_model,
                    persist_directory=DB_DIR
                )
                
                # Save reference to session state so we can close it later
                st.session_state["vectorstore"] = vectorstore
                
                # Clean up the temporary file
                os.remove(temp_pdf_path)
                
                st.success("Knowledge base built successfully! You can now ask questions.")
                st.session_state["db_ready"] = True
    elif process_button and uploaded_file is None:
        st.error("Please upload a PDF file first.")

# --- Main Chat Interface ---
# Check if vector store exists or has been built in this session
if os.path.exists(DB_DIR) or st.session_state.get("db_ready", False):
    
    # Initialize retriever using active session or load a new reference
    if st.session_state.get("vectorstore") is None:
        st.session_state["vectorstore"] = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
        
    retriever = st.session_state["vectorstore"].as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5}
    )
    
    # Prompt Template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant.
        Use only the provided context to answer the question. If you don't know the answer, say you don't know. Always use all the provided context to answer the question."""),
        ("human", "context: {context}\n\nquestion: {question}")
    ])

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display historical chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if user_query := st.chat_input("Ask a question about your book..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate response using RAG pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching book context..."):
                # Retrieve relevant document chunks
                docs = retriever.invoke(user_query)
                context = "\n\n".join([f"{doc.page_content} (Source: {doc.metadata.get('source', 'Uploaded Book')})" for doc in docs])
                
                # Format prompt and invoke LLM
                final_prompt = prompt_template.invoke({
                    "context": context,
                    "question": user_query
                })
                
                response = llm.invoke(final_prompt)
                ai_answer = response.content
                
                # Display response
                st.markdown(ai_answer)
                
                # Expandable section to peek at what chunks were used
                with st.expander("🔍 View Retrieved Sources"):
                    for i, doc in enumerate(docs):
                        st.caption(f"**Chunk {i+1}**:")
                        st.write(doc.page_content)
                        
        st.session_state.messages.append({"role": "assistant", "content": ai_answer})

else:
    st.info("👈 Please upload a PDF book in the sidebar and click 'Process' to start chatting!")