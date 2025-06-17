# narrative_agent.py

import pandas as pd

def generate_narratives(data: pd.DataFrame) -> list:
    """
    Generate business-style insights from supply chain data.
    """
    insights = []

    # --- 1. Overall Risk Summary ---
    total_flags = len(data)
    avg_cost = data['cost_impact'].mean()
    insights.append(
        f"⚠️ A total of {total_flags} supply chain records have been flagged for risk, "
        f"with an average financial impact of ₹{avg_cost:,.2f}."
    )

    # --- 2. High Delay Contributors ---
    high_delays = data[data['delay_days'] > 7]
    if not high_delays.empty:
        top_delayer = high_delays['supplier_name'].value_counts().idxmax()
        insights.append(
            f"📌 Supplier '{top_delayer}' is frequently associated with delays over 7 days."
        )

    # --- 3. High Cost Impact ---
    high_costs = data.sort_values(by='cost_impact', ascending=False).head(3)
    for _, row in high_costs.iterrows():
        insights.append(
            f"💸 Invoice {row['invoice_id']} from {row['supplier_name']} has a high cost impact "
            f"of ₹{row['cost_impact']:,.2f} with a delay of {row['delay_days']} days."
        )

    # --- 4. Region & Category Analysis ---
    if 'region' in data.columns and 'category' in data.columns:
        top_region = data['region'].value_counts().idxmax()
        top_category = data['category'].value_counts().idxmax()
        insights.append(
            f"🗺️ The region most affected is '{top_region}', especially in the '{top_category}' category."
        )

    return insights
