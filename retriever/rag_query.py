# retriever/rag_query.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def query_rag(user_question):
    # Initialize the embeddings (use the same model as rag_pipeline.py)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the saved FAISS vector store
    vector_db = FAISS.load_local("retriever/faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search
    results = vector_db.similarity_search(user_question, k=3)

    print("\n🔍 Top Relevant Chunks:\n")
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content)
        print()

# Run the script directly
if __name__ == "__main__":
    print("🧠 SupplyChainGPT - RAG Query")
    user_question = input("\nEnter your query: ")
    query_rag(user_question)
