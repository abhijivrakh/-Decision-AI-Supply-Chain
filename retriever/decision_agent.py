import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from rules_engine import check_delay_risk, get_flagged_records
from insight_engine import generate_insights
from rules_engine import get_flagged_records
from narrative_agent import generate_narratives






# Load data
csv_path = "supplier_invoices_large.csv"
df = pd.read_csv(csv_path)
print("Data Loaded!")
print(df.head())

# Load Vector DB
print(" Loading Vector Database...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("retriever/faiss_index", embedding_model, allow_dangerous_deserialization=True)
print("Vector DB Loaded!")
print("\n--- Running Rule Engine Checks ---")
flagged = get_flagged_records(df)
if not flagged.empty:
    print("⚠️ Flagged Records:")
    print(flagged[['invoice_id', 'supplier_name', 'delay_days', 'cost_impact']])
else:
    print("No critical delays or cost risks found.")

# Load lightweight HF model for fast testing
hf_pipeline = pipeline("text2text-generation", model="t5-small")
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Custom prompt template
prompt_template = """
You are a supply chain assistant. Based on the context below, answer the user question clearly and helpfully.

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# Ask a sample question
user_query = "Which supplier caused the highest cost impact due to delays?"
def classify_intent(query: str):
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ["insight", "region", "top category", "summary", "overall", "which category"]):
        return "insight"
    elif any(keyword in query_lower for keyword in ["delayed invoice", "delay", "rule", "flagged", "violation", "supplier delay", "threshold"]):
        return "rule"
    else:
        return "rag"

intent = classify_intent(user_query)
print(f"📌 Detected Intent: {intent}")

if intent == "rule":
    flagged = get_flagged_records(df)
    response = f"⚠️ Flagged Records:\n{flagged.to_string(index=False)}"
    narratives = generate_narratives(flagged)
    print("\n🧠 Narrative Summary from Rules:")
    for i, line in enumerate(narratives, 1):
        print(f"{i}. {line}")
elif intent == "insight":
    insights = generate_insights(df)
    response = "\n".join(insights)
else:
    response = qa_chain.run(user_query)

print("\n🔍 User Query:", user_query)
print("📢 Assistant Response:", response)

print("\n--- Generating Strategic Insights ---")
insights = generate_insights(df)
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")
