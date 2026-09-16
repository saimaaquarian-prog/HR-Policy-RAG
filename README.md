# 📋 HR Policy Assistant — RAG Application

An AI-powered HR Policy Assistant that allows employees to upload an HR policy PDF and ask questions about the document.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant sections from the HR policy before generating an answer with a Large Language Model.

---

## 🚀 Features

- Upload HR policy PDF
- Extract PDF text using PyMuPDF
- Split document into chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS
- Retrieve relevant policy sections
- Generate answers using Groq
- Uses `openai/gpt-oss-20b`
- Shows retrieved policy sources
- Displays policy page numbers
- Chat history for previous questions
- Simple Streamlit interface
- No database server required

---

## 🧠 RAG Architecture

```text
                 USER
                   |
                   v
          Upload HR Policy PDF
                   |
                   v
              PyMuPDF
                   |
                   v
           Extract PDF Text
                   |
                   v
              Text Chunks
                   |
                   v
        Sentence Transformer
                   |
                   v
             Embeddings
                   |
                   v
               FAISS
          Vector Database
                   |
                   |
       -------------------------
       |                       |
       |       User Question   |
       |              |        |
       |              v        |
       |      Question Embedding
       |              |
       |              v
       |         FAISS Search
       |              |
       |              v
       |    Relevant Policy Chunks
       |              |
       -------------------------
                   |
                   v
          Groq LLM API
       openai/gpt-oss-20b
                   |
                   v
            Grounded Answer
                   |
                   v
                 USER
