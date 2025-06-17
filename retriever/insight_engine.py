def generate_insights(data):
    """
    Analyzes supply chain invoice data and returns strategic insights as a dictionary,
    such as most delayed supplier, highest cost region, most affected category, etc.
    """
    insights = {}

    # Filter only delayed invoices
    delayed = data[data["delay_days"] > 0]

    if not delayed.empty:
        # Most delayed supplier (based on number of delayed invoices)
        delay_counts = delayed.groupby("supplier_name").size()
        top_supplier = delay_counts.idxmax()
        top_delay_count = delay_counts.max()
        insights["most_delayed_supplier"] = f"{top_supplier} ({top_delay_count} delayed invoices)"

        # Supplier with highest total cost impact due to delays
        cost_by_supplier = delayed.groupby("supplier_name")["cost_impact"].sum()
        top_cost_supplier = cost_by_supplier.idxmax()
        top_cost = cost_by_supplier.max()
        insights["supplier_with_highest_cost_impact_due_to_delay"] = f"{top_cost_supplier} (₹{top_cost:.2f})"

        # Region with highest total cost impact
        if "region" in data.columns:
            cost_by_region = delayed.groupby("region")["cost_impact"].sum()
            top_region = cost_by_region.idxmax()
            region_cost = cost_by_region.max()
            insights["region_with_highest_cost_impact"] = f"{top_region} (₹{region_cost:.2f})"

        # Category with the most delay days
        if "category" in data.columns:
            delay_by_category = delayed.groupby("category")["delay_days"].sum()
            top_category = delay_by_category.idxmax()
            delay_days = delay_by_category.max()
            insights["category_most_impacted_by_delays"] = f"{top_category} ({delay_days} delay days)"
    else:
        insights["no_delays_found"] = "✅ No delayed invoices were found in the dataset."

    return insights
