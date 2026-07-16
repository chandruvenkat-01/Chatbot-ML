# GenAI-Projects
# 📄 DocuMind — Chat With Your PDFs

> Ask questions. Get answers grounded in your document, not the internet.

DocuMind is a lightweight Retrieval-Augmented Generation (RAG) chatbot that reads a PDF, breaks it into searchable chunks, embeds them into a vector database, and lets an LLM answer your questions using **only** what's actually in the document — no guessing, no outside knowledge, no hallucinated citations.

Built as a hands-on exploration of the full RAG pipeline: text extraction → NLP preprocessing → chunking → embedding → vector search → grounded generation.

---

## 🧠 How It Works

```
PDF File
   │
   ▼
┌──────────────────┐
│  Text Extraction  │  → PyPDF pulls raw text from every page
└──────────────────┘
   │
   ▼
┌──────────────────┐
│  NLP Preprocessing │ → lowercase, strip HTML/punctuation/numbers,
└──────────────────┘    tokenize, remove stopwords, lemmatize
   │
   ▼
┌──────────────────┐
│   Chunking        │ → RecursiveCharacterTextSplitter
└──────────────────┘    (500 chars/chunk, 50 char overlap)
   │
   ▼
┌──────────────────┐
│   Embedding        │ → sentence-transformers/all-MiniLM-L6-v2
└──────────────────┘
   │
   ▼
┌──────────────────┐
│  Vector Store       │ → FAISS (saved locally, reloadable)
└──────────────────┘
   │
   ▼
┌──────────────────┐
│  Similarity Search │ → top-k most relevant chunks for your question
└──────────────────┘
   │
   ▼
┌──────────────────┐
│   LLM Answer        │ → Groq (Llama 3.3 70B) answers strictly
└──────────────────┘    from the retrieved context
```

---

## ✨ Features

- 📖 **Any PDF, instantly searchable** — drop in a document, no manual tagging or indexing required
- 🔍 **Semantic search, not keyword search** — finds relevant passages even when your question doesn't match the exact wording
- 💾 **Persistent vector index** — embed once, query forever (FAISS index saved to disk)
- 🚫 **Context-locked answers** — the LLM is explicitly instructed to answer only from retrieved content, reducing hallucination
- ⚡ **Fast inference** — powered by Groq's LPU inference engine

---

## 🛠️ Tech Stack

| Layer            | Tool                                             |
|-------------------|---------------------------------------------------|
| PDF Parsing       | [`pypdf`](https://pypi.org/project/pypdf/)        |
| Text Cleaning     | `BeautifulSoup`, `re`, `string`                    |
| NLP Preprocessing | `nltk` (tokenization, stopwords, lemmatization)    |
| Chunking          | `langchain_text_splitters`                         |
| Embeddings        | `sentence-transformers` (`all-MiniLM-L6-v2`)       |
| Vector Store      | `FAISS` via `langchain_community`                  |
| LLM               | `langchain_groq` → `llama-3.3-70b-versatile`       |

---

## 📁 Project Structure

```
.
├── stat.ipynb          # Main notebook — the full RAG pipeline, step by step
├── power bi.pdf         # Example source document (swap with your own)
├── power_index/         # Saved FAISS vector index (generated on first run)
├── .env                  # Your API keys (create this — see Setup below)
└── README.md
```

---

## 🚀 Setup

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/documind.git
cd documind
```

**2. Install dependencies**
```bash
pip install pypdf beautifulsoup4 nltk langchain-text-splitters \
    sentence-transformers langchain-community faiss-cpu langchain-groq python-dotenv
```

**3. Download NLTK data (one-time)**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

**4. Set your Groq API key — never hardcode it**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) — no credit card required.

Then in the notebook, load it instead of pasting it directly:
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

**5. Run the notebook**

Replace `power bi.pdf` with your own document, run all cells top to bottom, and start asking questions at the `ques = '...'` cell.

---

## 💬 Example

```python
ques = "define powerbi"
search = vector_db.similarity_search(ques, k=4)
```

```
Response:
Power BI is a business analytics service by Microsoft that provides
interactive visualizations and business intelligence capabilities...
```

---

## 🗺️ Roadmap

- [ ] Wrap the pipeline in a simple Streamlit or Gradio UI
- [ ] Add a "not found in document" fallback to further reduce hallucination
- [ ] Support multi-PDF ingestion into a single index
- [ ] Swap raw lemmatized text for original sentences at the embedding stage (better semantic fidelity)
- [ ] Add answer citations (which chunk/page supported the response)
- [ ] Dockerize for one-command setup

---

## ⚠️ Notes on API Keys

This repo intentionally does **not** commit any API keys. If you fork or clone this project, generate your own free Groq key and store it in a local `.env` file (already gitignored). Never commit `.env` or paste live keys into notebooks that get pushed to a public repo.

---

## 📜 License

MIT — free to use, modify, and build on.

---

## 🙋 Why "DocuMind"?

Because it's not just retrieving text — it's holding the document's context in mind, and refusing to answer beyond it. That constraint is the entire point of RAG: useful, grounded, and honest about what it doesn't know.
