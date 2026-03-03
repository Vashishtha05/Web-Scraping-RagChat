# 🧠 Multi-PDF RAG Chatbot — Retrieval Augmented Generation System

<p align="center">
  <img src="https://img.shields.io/badge/GenAI-RAG-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/LLM-Context%20Aware-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Web%20Scraping-Data%20Pipeline-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-LLM%20App-yellow?style=for-the-badge">
</p>

<p align="center">
  A Retrieval-Augmented Generation (RAG) chatbot that processes multiple PDFs and web data to deliver accurate, context-aware responses.
</p>

---

## 📌 Overview

This project implements a **RAG-based conversational AI system** that combines:

- Document retrieval from multiple PDFs  
- Web-scraped knowledge sources  
- Large Language Model (LLM) response generation  

The system enhances search-based interactions by generating precise, contextual answers grounded in retrieved content.

---

## 🎯 Objective

- Provide fast, accurate answers from large document collections  
- Reduce manual search effort across PDFs  
- Enable context-aware conversational responses  
- Improve decision-making using structured knowledge retrieval  

---

## ⚙️ How It Works

### 1️⃣ Document Ingestion
- Multiple PDFs are uploaded  
- Text is extracted and chunked  
- Embeddings are generated and stored in a vector database  

### 2️⃣ Retrieval
- User query is converted into embeddings  
- Similar document chunks are retrieved using vector similarity search  

### 3️⃣ Generation
- Retrieved context is passed to an LLM  
- The LLM generates a coherent, context-grounded response  

---

## 🏗️ Architecture Flow

User Query  
→ Embedding Model  
→ Vector Search (Relevant Chunks)  
→ LLM with Retrieved Context  
→ Final Response  

---

## ✨ Features

- 📄 Multi-PDF document support  
- 🌐 Web scraping integration  
- 🔍 Vector similarity search  
- 🤖 Context-aware LLM responses  
- ⚡ Fast retrieval pipeline  
- 🧠 Scalable RAG architecture  

---

## 🛠 Tech Stack

| Technology | Usage |
|------------|--------|
| Python | Core development |
| LLM (GPT / compatible model) | Response generation |
| Embedding Model | Vector representations |
| FAISS / Vector DB | Similarity search |
| Web Scraping Tools | Knowledge extraction |
| Streamlit (optional) | UI interface |

---

## 🚀 Getting Started

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/multi-pdf-rag-chatbot.git
cd multi-pdf-rag-chatbot
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application

```bash
python app.py
```

---

## 📂 Project Structure

```
├── app.py
├── pdf_loader.py
├── web_scraper.py
├── embeddings.py
├── vector_store.py
├── rag_pipeline.py
├── requirements.txt
└── README.md
```
## 👨‍💻 Author 
**Vashishtha Verma** 
* 🤖 Machine Learning & Generative AI
* 🧠 Agentic AI Systems
* 💻 Software Engineering & DSA
