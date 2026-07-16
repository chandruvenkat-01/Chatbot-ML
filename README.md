<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=C8A2FF&height=220&section=header&text=ML%20Notes%20PDF%20Chatbot&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=GenAI%20|%20NLP%20|%20FAISS%20|%20Groq%20LLM&descAlignY=58&descColor=ffffff&descSize=18"/>


# 🤖 ML Notes PDF Chatbot

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-00C853?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-4285F4?style=for-the-badge)
![Sentence Transformers](https://img.shields.io/badge/Sentence--Transformers-FF6F00?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge)

### 🚀 Ask Your ML Notes Questions, Get Grounded, Context-Aware Answers

</div>

---

# 📖 Project Overview

This project is an **AI-powered PDF Chatbot** that lets you ask natural language questions about a Machine Learning notes PDF and get answers pulled directly from the document — not from the model's general training knowledge.

The pipeline extracts text from the PDF, cleans and normalizes it with classic NLP techniques, splits it into overlapping chunks, embeds those chunks into a **FAISS vector database**, retrieves the most relevant chunks for a given question, and passes them to an LLM (via **Groq**) with strict instructions to answer only from that retrieved context.

This project is a hands-on implementation of the core ideas behind **Retrieval-Augmented Generation (RAG)**.

---

# 🎯 Objective

Build an intelligent chatbot capable of answering questions directly from a PDF of ML notes using:

- 📄 PDF Text Extraction
- 🧠 Natural Language Processing (tokenization, stopword removal, lemmatization)
- 🔍 Semantic Search via Sentence Embeddings
- ⚡ FAISS Vector Database
- 🗣️ Groq-hosted LLM (Llama 3.3 70B) for grounded answer generation

---

# 📂 Project Structure

```text
ML-Notes-Chatbot/
│
├── ML nots.pdf          # Source PDF (ML notes)
├── chat.ipynb            # Full pipeline notebook — extraction to answer
├── faiss_index/          # Saved FAISS vector index (generated on first run)
├── .env                    # API keys (create this — see Setup below)
└── README.md
```

---

# 📄 Dataset

Instead of a traditional structured dataset, this project uses a **Machine Learning notes PDF** as its knowledge source. The chatbot retrieves and answers strictly from this document's content.

---

# ⚙️ Workflow

```text
PDF Document
      │
      ▼
Extract Text (PyPDF)
      │
      ▼
Text Cleaning (lowercase, HTML strip, punctuation/number removal)
      │
      ▼
Tokenization → Stopword Removal → Lemmatization
      │
      ▼
Chunking (RecursiveCharacterTextSplitter, 500 chars, 50 overlap)
      │
      ▼
Sentence Embeddings (all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Database
      │
      ▼
User Question
      │
      ▼
Similarity Search (Top-K)
      │
      ▼
Retrieved Context
      │
      ▼
Groq LLM (Llama 3.3 70B) → Grounded Answer
```

---

# 🧠 Technologies Used

- Python
- PyPDF
- BeautifulSoup
- NLTK
- LangChain (`langchain-text-splitters`, `langchain-community`, `langchain-groq`)
- Sentence Transformers
- FAISS
- Groq API
- Jupyter Notebook

---

# 🤖 Core Components

### PDF Processing
- Extract Text
- Clean Text
- Normalize Content

### Text Processing
- Tokenization
- Stopword Removal
- Lemmatization
- Chunking

### Semantic Search
- Sentence Embeddings
- FAISS Similarity Search
- Top-K Retrieval

### Answer Generation
- Context-Restricted Prompting
- Groq LLM Inference (Llama 3.3 70B)

---

# 📝 NLP Pipeline

- PDF Text Extraction
- Lowercase Conversion
- HTML/Punctuation/Number Stripping
- Whitespace Normalization
- Tokenization
- Stopword Removal
- Lemmatization
- Chunk Creation
- Sentence Embedding Generation
- Vector Index Creation
- Similarity Search
- Context-Grounded LLM Response

---

# 📊 Workflow Architecture

```text
User Question
      │
      ▼
Embedding Model (all-MiniLM-L6-v2)
      │
      ▼
FAISS Search
      │
      ▼
Top-K Similar Chunks
      │
      ▼
Retrieved Context
      │
      ▼
Groq LLM (Llama 3.3 70B)
      │
      ▼
Final Answer
```

---

# ✨ Features

- ✅ PDF Question Answering
- ✅ Full NLP Preprocessing Pipeline
- ✅ Semantic (Not Just Keyword) Search
- ✅ FAISS Vector Database with Local Persistence
- ✅ Sentence Embeddings via Sentence-Transformers
- ✅ Context-Restricted Prompting to Reduce Hallucination
- ✅ Fast Inference via Groq
- ✅ Scalable, Modular Pipeline Design

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/ml-notes-chatbot.git
```

Move into the project

```bash
cd ml-notes-chatbot
```

Install dependencies

```bash
pip install pypdf beautifulsoup4 nltk langchain-text-splitters \
    sentence-transformers langchain-community faiss-cpu langchain-groq python-dotenv
```

Download required NLTK data (one-time)

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

# 🔑 API Key Setup

This project uses the **Groq API** for LLM inference. Get a free key at [console.groq.com](https://console.groq.com) — no credit card required.

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Then load it in the notebook instead of hardcoding it:

```python
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)
```

> ⚠️ Never commit a real API key to GitHub. Keep `.env` in your `.gitignore`.

---

# ▶️ Run the Project

```bash
jupyter notebook chat.ipynb
```

Run all cells top to bottom, then edit the `question` variable near the end to ask anything about the source PDF:

```python
question = 'What is multi-class classification'
```

---

# 💬 Example

```python
question = "What is multi-class classification"
search = vector_db.similarity_search(question, k=5)
```

```
Response:
Multi-class classification is a classification task with more than two
classes, where each input is assigned to one of several possible
categories based on the patterns learned from the training data...
```

---

# 🛠 Skills Demonstrated

- Generative AI Fundamentals
- Retrieval-Augmented Generation (RAG)
- Natural Language Processing
- Semantic Search
- Vector Databases
- FAISS
- Sentence Transformers
- Text Chunking
- PDF Processing
- Prompt Engineering
- Python

---

# 🔮 Future Improvements

- [ ] Fallback response when the answer isn't found in the document
- [ ] Preserve original sentence structure for embeddings (skip lemmatization at embedding stage for better semantic fidelity)
- [ ] Streamlit or Gradio web interface
- [ ] Multi-PDF support with source/page citations
- [ ] Conversation memory for follow-up questions
- [ ] LangChain `RetrievalQA` chain refactor
- [ ] Docker deployment
- [ ] Flask REST API wrapper

---

# 📌 Applications

- Educational / Study Notes Assistant
- Research Paper Q&A
- Company Knowledge Base Search
- Technical Documentation Chatbot
- Digital Library Search
- AI Learning Companion

---

# 🤝 Contributing

Contributions are welcome!

1. Fork this repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 👨‍💻 Author

## [Your Name]

**GenAI & NLP Enthusiast | Building Retrieval-Augmented Applications**

📊 Exploring the intersection of NLP, vector search, and LLM-powered applications.

💡 Building intelligent AI solutions using Python, FAISS, LangChain, and Groq.

🌱 Currently learning advanced RAG architectures, LangChain pipelines, and LLM evaluation.

---

<div align="center">

## ⭐ If you found this project useful, please give it a Star!

### 💬 "Turning static PDFs into conversational, context-aware knowledge assistants."

<img src="https://capsule-render.vercel.app/api?type=waving&color=C8A2FF&height=120&section=footer"/>

</div>
