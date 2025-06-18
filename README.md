# 📦 SupplyChainGPT: A GenAI-Powered Supply Chain Assistant

SupplyChainGPT is an end-to-end AI-powered assistant designed to support supply chain analysis using Retrieval-Augmented Generation (RAG), rule-based alerts, strategic insight generation, and data visualization. This app helps stakeholders understand supplier performance, delays, cost impacts, and optimization strategies—powered by LLMs, FAISS, and Streamlit.

---

## 🚀 Features

- ✅ Upload and analyze supplier invoice datasets (CSV)
- 🧠 RAG-based question answering using FAISS + HuggingFace
- 📊 Automatic strategic insights and visual dashboards
- ⚠️ Rule engine to flag delayed or high-cost invoices
- ✨ Smart narrative summaries for executive reporting
- 🎯 Custom logic to answer domain-specific business queries

---

## 🛠️ Tech Stack

| Layer           | Tools/Frameworks                                               |
|----------------|-----------------------------------------------------------------|
| Frontend       | Streamlit                                                      |
| LLM Pipeline   | HuggingFace Transformers (T5-small), LangChain                 |
| Embedding      | SentenceTransformers (`all-MiniLM-L6-v2`)                      |
| Vector DB      | FAISS (local index)                                            |
| Backend Logic  | Pandas, Matplotlib, Rules Engine, Insight Engine               |
| Deployment     | Anaconda + Python 3.10                                          |

---

## 🧩 Project Structure

