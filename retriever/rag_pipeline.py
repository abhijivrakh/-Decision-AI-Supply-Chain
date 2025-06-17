# retriever/rag_pipeline.py

import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_and_embed_data(csv_path):
    print(" Loading CSV...")
    loader = CSVLoader(file_path=csv_path)
    documents = loader.load()

    print(" Splitting text...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = splitter.split_documents(documents)

    print("Initializing HuggingFace Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("📦 Creating FAISS vector DB...")
    vector_db = FAISS.from_documents(split_texts, embeddings)

    # ✅ Save for later use
    print(" Saving FAISS index locally...")
    vector_db.save_local("retriever/faiss_index")

    print("Embedding and vector DB created + saved successfully.")

# Call the function only when running this script directly
if __name__ == "__main__":
    csv_file = "supplier_invoices_large.csv"  # Make sure this path is correct
    load_and_embed_data(csv_file)
