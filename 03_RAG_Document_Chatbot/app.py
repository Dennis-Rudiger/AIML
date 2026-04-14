import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# NOTE for Beginners: We are using HuggingFace for Embeddings because it's FREE and runs on your CPU.
# If you don't have an OpenAI API key, this script includes a "Mock Mode" that simulates 
# an AI answering so you can still present the core RAG architecture (LangChain + Chroma Vector DB).

st.set_page_config(page_title="RAG Document Chatbot", page_icon="🤖", layout="wide")

# --- UI Setup ---
st.title("🤖 Chat with Your Documents (RAG)")
st.write("Upload a TXT file and ask questions about its content. This app uses Retrieval-Augmented Generation (RAG) to find exact answers.")

# Sidebar for settings/file upload
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("OpenAI API Key (leave empty for Mock Mode)", type="password")
    uploaded_file = st.file_uploader("Upload a text file to chat with (.txt)", type=("txt"))
    st.markdown("---")
    st.write("📝 **Pro Tip:** Use the provided `sample_policy.txt` to test the WFH / PTO rules.")

# Initialize the chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Upload a document and ask me anything about it."}]

# Initialize the vector store in session state so we don't recreate it on every rerun
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Core RAG Pipeline (Data Preprocessing & Storage) ---
def process_document(file):
    # 1. Save uploaded file temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.name)
    
    with open(temp_file_path, "wb") as f:
        f.write(file.getvalue())
        
    st.sidebar.success(f"Processing '{file.name}'...")
    
    # 2. Load the Document (Read the text)
    loader = TextLoader(temp_file_path, encoding="utf-8")
    documents = loader.load()
    
    # 3. Chunk the Document (LLMs can't read 10,000 pages at once. We cut it into 500-letter paragraphs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    st.sidebar.write(f"- Cut into {len(chunks)} chunks.")
    
    # 4. Embeddings (Turn the text chunks into math/numbers)
    # We use a free, lightweight model from HuggingFace to turn text into vectors
    # all-MiniLM-L6-v2 is an excellent starter model for semantic similarity
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 5. Save to Vector Database (ChromaDB)
    # We overwrite the DB in memory for this session
    st.sidebar.write("- Creating Vector Database...")
    vector_store = Chroma.from_documents(chunks, embeddings)
    st.session_state.vector_store = vector_store
    
    st.sidebar.success("✅ Database Ready! You can chat now.")
    
    # Cleanup temp file
    try:
        os.remove(temp_file_path)
    except:
        pass

# Trigger processing if a new file is uploaded
if uploaded_file and st.session_state.vector_store is None:
    with st.spinner("Building the RAG Pipeline..."):
        process_document(uploaded_file)

# --- Chat Interface & Answer Generation ---
# Display previous chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input
if prompt := st.chat_input("E.g., How many vacation days do I get?"):
    
    # Show user message
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # The Bot thinks...
    if st.session_state.vector_store is None:
        response = "Please upload a document to the sidebar first!"
    else:
        # ** THE MAGIC OF RAG HAPPENS HERE **
        
        # Step 1: Retrieval (Search the Vector Database for paragraphs matching the user's question)
        docs = st.session_state.vector_store.similarity_search(prompt, k=2) # Get top 2 chunks
        
        # Combine the retrieved text
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Step 2: Generation (Feed the retrieved context + the question to an LLM)
        if api_key_input.startswith("sk-"):
            # REAL LLM MODE (Requires openAI key)
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            
            try:
                llm = ChatOpenAI(temperature=0, openai_api_key=api_key_input)
                # We enforce the rules so the AI doesn't hallucinate
                system_prompt = (
                    "You are a helpful assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the user's question. "
                    "If you don't know the answer, just say 'According to the document, I do not know.' "
                    "\n\nContext:\n{context}"
                )
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{question}")
                ])
                chain = prompt_template | llm
                
                with st.spinner("Reading the vector DB and generating answer..."):
                    ai_reply = chain.invoke({"context": context_text, "question": prompt})
                    response = ai_reply.content
            except Exception as e:
                response = f"OpenAI API Error: {str(e)}"
                
        else:
            # MOCK MODE (For beginners running this locally without API keys)
            # We skip the LLM generation but we still PROVE the Vector Retreival worked
            response = (
                f"*(MOCK MODE)* Based on my semantic search of your document, here is the relevant section I found:\n\n"
                f"**> {context_text}**\n\n"
                f"*(Note: Provide an OpenAI API key in the sidebar to have an AI natively summarize this into a human response!)*"
            )

    # Show AI response
    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
