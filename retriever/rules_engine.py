# rules_engine.py
import pandas as pd
def check_delay_risk(record):
    """Flag if the delay is critical (e.g., >7 days)"""
    return record['delay_days'] > 7

def check_cost_impact(record):
    """Flag if cost impact is above defined threshold"""
    return record['cost_impact'] > 5000

def check_supplier_performance(dataframe):
    """Return suppliers with high average delay"""
    return (
        dataframe[dataframe['delay_days'] > 0]
        .groupby('supplier_name')['delay_days']
        .mean()
        .sort_values(ascending=False)
    )

def get_flagged_records(dataframe):
    """Return all records that meet one or more risk conditions"""
    return dataframe[
        (dataframe['delay_days'] > 7) | (dataframe['cost_impact'] > 5000)
    ]
def send_alerts_for_flags(flagged_df: pd.DataFrame):
    """
    Send alerts for critical flags. Currently simulates by printing alerts.
    """
    for _, row in flagged_df.iterrows():
        alert_msg = (
            f"[ALERT] 🚨 Invoice {row['invoice_id']} from {row['supplier_name']} "
            f"has a delay of {row['delay_days']} days and cost impact of ₹{row['cost_impact']}."
        )
        print(alert_msg)  # You can later replace this with email/SMS/Slack API
