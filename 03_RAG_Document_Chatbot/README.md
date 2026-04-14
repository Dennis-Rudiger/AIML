# Project 3: RAG Document Chatbot 🤖📄

This is the most advanced and highly sought-after project in your portfolio: **Retrieval-Augmented Generation (RAG)**. 

## The Goal
To build a web application (using Streamlit) where a user can upload a text document or PDF, and then "chat" with it. Instead of hallucinating, the AI will search the document for the exact answer and generate a response based *only* on your private data.

## Skills Demonstrated
*   **Frameworks:** `LangChain`, `Streamlit` (for the Web UI)
*   **Generative AI Concepts:** 
    *   **Embeddings:** Turning text into numbers (vectors) so the computer can understand meaning.
    *   **Vector Databases:** Using `ChromaDB` to store and search the document chunks.
    *   **LLM Orchestration:** Connecting a Large Language Model (LLM) to a private data source.

## 📁 Files Explained
1.  **`app.py`**: The main web application. It handles the user interface, processes the uploaded document, chunks it into pieces, saves it to a vector database, and handles the chat logic.
2.  **`sample_policy.txt`**: A fake company document you can use to test the chatbot.
3.  **`.env.example`**: A template for storing your API keys securely.

## 🚀 How to Run (Beginner Friendly)

**Step 1:** Install the required Generative AI libraries:
```powershell
pip install streamlit langchain langchain-openai chromadb sentence-transformers
```

**Step 2:** Add your API Key
Rename `.env.example` to `.env` and add your OpenAI API key. *(Note: The code includes a "Mock Mode" that simulates the AI if you don't have an API key right now!)*

**Step 3:** Start the Web App!
```powershell
streamlit run app.py
```
*(This will automatically open a beautiful webpage in your browser).*