import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from rules_engine import get_flagged_records, send_alerts_for_flags
from insight_engine import generate_insights
from narrative_agent import generate_narratives

# Set up Streamlit UI
st.set_page_config(page_title="SupplyChainGPT", layout="wide")
st.title("📦 Supply Chain Assistant with GenAI")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Supplier Invoice CSV", type=["csv"])

def custom_logic_handler(query: str, df: pd.DataFrame):
    query_lower = query.lower()

    if "most delays" in query_lower:
        delay_counts = df[df["delay_days"] > 0].groupby("supplier_name").size()
        if not delay_counts.empty:
            top_supplier = delay_counts.idxmax()
            top_count = delay_counts.max()
            return f"🕒 {top_supplier} has the most delayed invoices ({top_count} delays)."
        else:
            return "✅ No delayed invoices found."

    elif "highest cost impact" in query_lower:
        max_cost_row = df[df["delay_days"] > 0].sort_values("cost_impact", ascending=False).iloc[0]
        return (f"💸 {max_cost_row['supplier_name']} caused the highest cost impact "
                f"of ₹{max_cost_row['cost_impact']:.2f} due to delays (Invoice ID: {max_cost_row['invoice_id']}).")

    elif "average delay" in query_lower:
        for supplier in df["supplier_name"].unique():
            if supplier.lower() in query_lower:
                avg_delay = df[df["supplier_name"] == supplier]["delay_days"].mean()
                return f"📊 Average delay for {supplier} is {avg_delay:.2f} days."
        return "❓ Please specify a valid supplier name to calculate average delay."
    elif "improve supply chain" in query_lower:
        total_invoices = len(df)
        delayed = df[df["delay_days"] > 0]
        delay_rate = (len(delayed) / total_invoices) * 100
        top_regions = delayed.groupby("region")["cost_impact"].sum().sort_values(ascending=False).head(2)
        suggestions = [
        f"🚀 Reduce delays in regions like {', '.join(top_regions.index)} to cut major cost impact.",
        f"📦 Focus on high-impact categories such as {', '.join(delayed['category'].value_counts().head(2).index)}.",
        f"⏱️ Currently, about {delay_rate:.2f}% of invoices are delayed — targeting a 10% reduction can save ₹{delayed['cost_impact'].sum():.2f}."
    ]
    return "\n".join(suggestions)


    return None  # Let LLM handle anything else

def show_visual_insights(df: pd.DataFrame):
    st.subheader("📊 Visual Insights")

    delay_counts = df[df["delay_days"] > 0].groupby("supplier_name").size().sort_values(ascending=False).head(5)
    if not delay_counts.empty:
        st.markdown("**Top 5 Most Delayed Suppliers**")
        st.bar_chart(delay_counts)

    cost_impact = df.groupby("supplier_name")["cost_impact"].sum().sort_values(ascending=False).head(5)
    if not cost_impact.empty:
        st.markdown("**Top 5 Suppliers by Cost Impact**")
        st.bar_chart(cost_impact)

    if "region" in df.columns:
        region_impact = df.groupby("region")["cost_impact"].sum()
        if not region_impact.empty:
            fig, ax = plt.subplots()
            ax.pie(region_impact, labels=region_impact.index, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.markdown("**Cost Impact Distribution by Region**")
            st.pyplot(fig)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data Uploaded Successfully!")
    st.write(df.head())

    st.info("🔍 Loading Vector Database...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("retriever/faiss_index", embedding_model, allow_dangerous_deserialization=True)
    st.success("✅ Vector DB Loaded!")

    hf_pipeline = pipeline("text2text-generation", model="t5-small")
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

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

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    user_query = st.text_input("Ask a Question:", "Which supplier caused the highest cost impact due to delays?")

    def classify_intent(query: str):
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in ["insight", "region", "top category", "summary", "overall", "which category"]):
            return "insight"
        elif any(keyword in query_lower for keyword in ["delayed invoice", "delay", "rule", "flagged", "violation", "supplier delay", "threshold"]):
            return "rule"
        else:
            return "rag"

    if st.button("Generate Response"):
        intent = classify_intent(user_query)
        st.write(f"📌 Detected Intent: `{intent}`")

        custom_response = custom_logic_handler(user_query, df)
        if custom_response:
            st.subheader("📌 Smart Response")
            st.success(custom_response)

        elif intent == "insight":
            insights = generate_insights(df)
            st.subheader("📊 Strategic Insights (Narrative)")
            for i, (k, v) in enumerate(insights.items(), 1):
                st.markdown(f"**{i}.** {k.replace('_', ' ').title()}: {v}")
            show_visual_insights(df)

        elif intent == "rule":
            flagged = get_flagged_records(df)
            if not flagged.empty:
                st.warning("⚠️ Flagged Records:")

                def highlight_issues(row):
                    if row["delay_days"] > 7 and row["cost_impact"] > 5000:
                        return ['background-color: #ff9999']*len(row)
                    elif row["delay_days"] > 7:
                        return ['background-color: #ffe0b3']*len(row)
                    elif row["cost_impact"] > 5000:
                        return ['background-color: #ffffb3']*len(row)
                    else:
                        return ['']*len(row)

                styled = flagged[['invoice_id', 'supplier_name', 'delay_days', 'cost_impact']].style.apply(highlight_issues, axis=1)
                st.dataframe(styled)

                send_alerts_for_flags(flagged)

                st.subheader("📖 Narrative Summary")
                narratives = generate_narratives(flagged)
                for i, line in enumerate(narratives, 1):
                    st.markdown(f"**{i}.** {line}")
            else:
                st.success("No critical delays or cost risks found.")

        else:
            response = qa_chain.run(user_query)
            st.subheader("🧜 RAG-based Answer")
            st.write(response)

        st.divider()
        st.subheader("📊 Full Strategic Insight Summary")
        insights = generate_insights(df)
        for i, (k, v) in enumerate(insights.items(), 1):
            st.markdown(f"**{i}.** {k.replace('_', ' ').title()}: {v}")
        show_visual_insights(df)
