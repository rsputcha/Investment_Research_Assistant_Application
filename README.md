# 📄 Investment Research Assistant — Full Project Report

---

## 🗂️ Project Overview

This project builds an **Investment Research Assistant** using **Retrieval-Augmented Generation (RAG)** principles.  
It helps investors and researchers **query financial datasets** (Earnings Call Transcripts, Financial News) and receive **relevant summarized answers**.

The final system includes:
- A **full backend pipeline** (Data Cleaning → Embedding → FAISS Storage → Hybrid Retrieval)
- A lightweight **demo frontend** using **Streamlit**.
- Free, open-source tech stack: **No paid APIs**, **no commercial LLMs**.
- Deployable on **Streamlit Community Cloud** or any local environment.

---

## 📊 Datasets Used

1. **US Financial News Articles**  
   - [Kaggle Link](https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles)  
   - ~27,000 news articles from major financial sources.

2. **Earnings Call Transcripts**  
   - [Kaggle Link](https://www.kaggle.com/datasets/ashwinm500/earnings-call-transcripts)  
   - Detailed transcripts from earnings calls of various public companies.

---

## 🏗️ Project Structure

```bash
investment-research-assistant/
├── app.py                  # Streamlit front-end (demo app)
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw/                 # Original downloaded datasets
│   └── processed/           # Cleaned and preprocessed files
├── embeddings/              # FAISS indexes and associated metadata
├── outputs/                 # Final RAG outputs, query logs
├── scripts/
│   └── preprocess_data.py   # Data cleaning and embedding generation scripts
├── assets/
│   └── screenshot.png       # UI screenshots, visual assets
├── README.md
└── REPORT.md                # Project report (this file)
```

---

## 🛠️ Technologies & Libraries

| Purpose                  | Tech Used                       |
|---------------------------|---------------------------------|
| Programming Language      | Python 3.10+                    |
| Vectorization (Embeddings) | `sentence-transformers` (MiniLM) |
| Vector Storage            | FAISS (Facebook AI Similarity Search) |
| Model for basic RAG in demo | TF-IDF + Cosine Similarity |
| Frontend                  | Streamlit                      |
| Data Handling             | Pandas, Numpy                  |
| Deployment (Demo App)     | Streamlit Community Cloud       |

---

## 🧠 Models Used

### 1. Main RAG Backend Model (Colab)
- **Sentence Transformer:** `all-MiniLM-L6-v2`
  - Source: Hugging Face `sentence-transformers`
  - Embedding size: 384 dimensions
  - Lightweight, fast, and effective for semantic search.
- **Vector Store:** FAISS
  - Flat Index (no clustering, just brute-force but efficient).

### 2. Streamlit Demo Version (Frontend App)
- Simplified retrieval: **TF-IDF + Cosine Similarity**
- Purpose: Lightweight demonstration without large model download.

---

## ⚙️ Backend Pipeline (Detailed)

1. **Data Preprocessing**
   - Remove nulls, non-English content, extremely short articles.
   - Normalize text (lowercase, remove special characters).
   - Separate fields: `company`, `date`, `content`, `source`.

2. **Embedding Generation**
   - Use `all-MiniLM-L6-v2` Sentence Transformer.
   - Create dense vector embeddings of each document.

3. **FAISS Indexing**
   - Build two separate FAISS indexes:
     - One for **Financial News Articles**
     - One for **Earnings Call Transcripts**
   - Store corresponding metadata (`company`, `date`, etc.).

4. **Hybrid Retrieval**
   - On user query:
     - Search both indexes.
     - Retrieve top-k relevant chunks.
     - Aggregate and prepare a response.

5. **Answer Generation (Colab Full Version)**
   - Combine retrieved documents into a context window.
   - Generate structured summaries (Template or LLM if available).
   - If LLM not available, fallback to pattern matching and concatenation.

---

## 🎯 Streamlit Demo Version (Simplified Flow)

- Loads **sample financial data** (fallback mode if real data unavailable).
- Builds **TF-IDF** vectors on sample documents.
- Accepts **user questions** through a simple text box.
- Retrieves top 3 most relevant documents.
- Uses **basic heuristics and pattern matching** to generate structured responses.
- Displays query **history** and **retrieved sources** interactively.

---

## 🚀 Deployment Instructions

**Run Locally:**
```bash
# Clone the repo
git clone https://github.com/yourusername/investment-research-assistant.git
cd investment-research-assistant

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

**Deploy on Streamlit Community Cloud:**
- Create a new app on [Streamlit Cloud](https://streamlit.io/cloud).
- Connect your GitHub repository.
- Set `app.py` as the main file.
- Deploy!

---

## 📸 Sample Screenshot

*(Screenshot is inside `assets/screenshot.png`)*

---

## 📈 Future Improvements

- Replace TF-IDF retrieval in frontend with real FAISS hybrid retriever.
- Connect to a lightweight local LLM (like `phi-2`, `mistral-7b-instruct`) using HuggingFace Inference.
- Implement real-time financial news scraping pipeline.
- Add simple ranking model to order results better (using BERTScore or reranking).
- Add caching mechanism for repeated queries.
- Improve frontend UI with charts/metrics using Streamlit-Extras.

---

## ✍️ Author

Created by **Putcha, Ram Srikar**  
For **Advanced Data Science Portfolio Project 2025**

---

# ✅ Summary

You now have:
- Full backend in Colab (real FAISS-based RAG system).
- Lightweight demo frontend (Streamlit, TF-IDF).
- Fully free and deployable project.
- Easy upgrade paths (from basic to real production-grade RAG).

---

