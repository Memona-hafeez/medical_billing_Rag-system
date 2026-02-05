# Medical Billing RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** based chatbot designed for **medical billing question answering**, using document retrieval, vector search, persistent memory, and cloud-backed storage via **Firebase**.

This system allows users to ask medical billing–related questions and receive **context-aware, document-grounded answers** instead of hallucinated responses.

---

## 📌 Project Overview

Medical billing involves strict rules, coding standards, and documentation requirements. Generic LLMs often fail in this domain due to lack of context.

This chatbot solves the problem by:

- Indexing medical billing documents
- Storing embeddings in a vector database (FAISS)
- Retrieving relevant content for every query
- Using persistent memory for conversation continuity
- Managing user/session data via **Firebase**

---

## ✨ Key Features

- Medical billing document ingestion
- Retrieval-Augmented Generation (RAG)
- FAISS vector store for semantic search
- Persistent memory creation and retrieval
- Firebase integration for backend storage
- Modular backend architecture
- Chat interface orchestration
- Designed to reduce hallucinations

---

## 🧠 Tech Stack

| Component | Technology |
|---------|------------|
| Language | Python |
| Vector Store | FAISS |
| RAG Framework | LangChain-style pipeline |
| Backend | Python (Flask-style logic) |
| Memory Storage | Firebase |
| Embeddings | OpenAI / compatible embedding model |
| Document Store | Local filesystem |
| Front Controller | `run.py` |

---

## 🏗️ System Architecture

```
Medical Billing Documents
           ↓
     Text Extraction
           ↓
      Text Chunking
           ↓
   Embedding Generation
           ↓
    FAISS Vector Store
           ↓
     Context Retrieval
           ↓
     LLM Inference
           ↓
    Contextual Answer
           ↓
   Firebase (Memory + State)
```

---

## 📁 Project Structure

```
Medical Billing Chatbot/
│
├── data/                       # Medical billing documents
│
├── vectorstore/
│   └── db_faiss/               # FAISS vector database files
│
├── app.py                      # Main chatbot interface logic
├── backend.py                  # Core RAG backend pipeline
├── create_memory.py            # Creates vector memory from documents
├── connect_memory.py           # Connects to stored FAISS memory
├── run.py                      # Application entry point
├── requirements.txt            # Project dependencies
├── .gitignore
└── README.md
```

---

## 🔍 File Responsibilities

### `app.py`
- Handles chatbot interaction flow
- Sends user queries to backend
- Returns generated answers

### `backend.py`
- Core RAG logic
- Combines retrieval + LLM inference
- Controls prompt construction and response generation

### `create_memory.py`
- Reads medical billing documents from `/data`
- Splits text into chunks
- Generates embeddings
- Stores vectors into FAISS database

### `connect_memory.py`
- Loads existing FAISS vector store
- Enables similarity search during inference
- Ensures memory persistence across sessions

### `run.py`
- Entry point for running the chatbot
- Initializes components and starts the app

---

## 🔥 Firebase Integration

Firebase is used for:

- Storing chat/session metadata
- Managing user memory or conversation history
- Persisting state across multiple interactions
- Supporting scalable backend operations

Firebase allows this chatbot to move beyond a local demo and toward a **production-ready architecture**.

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Memona-hafeez/medical_billing_Rag-system.git
cd medical_billing_Rag-system/Medical\ Billing\ Chatbot
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # Linux / Mac
venv\Scripts\activate        # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Create Vector Memory

Before running the chatbot, generate the vector memory:

```bash
python create_memory.py
```

This will:
- Read billing documents
- Generate embeddings
- Store vectors in `vectorstore/db_faiss`

---

## ▶️ Run the Chatbot

```bash
python run.py
```

The chatbot will now be ready to answer medical billing queries using retrieved context.

---

## 🧪 Example Queries

- What documentation is required for CPT 99214?
- How is ICD-10 code Z79.01 justified?
- What modifiers apply to outpatient billing?
- What are common billing denial reasons?

---

## ⚠️ Limitations

- Answers depend on available documents
- Not certified for legal or clinical decision-making
- Requires properly formatted billing data
- FAISS index must be regenerated when documents change

---

## 🚀 Future Improvements

- Add web-based UI (Streamlit / React)
- Source citation in answers
- Multi-user authentication via Firebase Auth
- Support for additional medical datasets
- Dockerized deployment

---

## 📜 Disclaimer

This chatbot is intended for **educational and assistive purposes only**.  
It does not replace certified medical billing professionals.

---

## 📄 License

Open-source project.  
Add an MIT or Apache-2.0 LICENSE before production use.
