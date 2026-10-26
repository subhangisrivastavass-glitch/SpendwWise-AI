def compare_with_benchmark(df, monthly_budget):
    benchmarks = {
        "Food": 8000,
        "Transport": 3000,
        "Entertainment": 2500,
        "Shopping": 5000,
        "Health": 2000,
        "Utilities": 2500,
        "Education": 2000,
        "Investment": 5000,
        "Rent": 15000,
    }

    results = []

    if "Month" in df.columns:
        months = max(df["Month"].nunique(), 1)
    else:
        months = 1

    for category, benchmark in benchmarks.items():
        user_spend = df[df["Category"] == category]["Amount"].sum() / months

        diff_pct = (
            ((user_spend - benchmark) / benchmark) * 100
            if benchmark > 0
            else 0
        )

        if diff_pct > 10:
            status = "above"
        elif diff_pct < -10:
            status = "below"
        else:
            status = "on_par"

        results.append({
            "category": category,
            "user_monthly": round(user_spend, 2),
            "benchmark_monthly": benchmark,
            "diff_pct": round(diff_pct, 2),
            "status": status,
        })

    return results


def benchmark_summary_line(benchmark_data):
    above = sum(1 for x in benchmark_data if x["status"] == "above")
    below = sum(1 for x in benchmark_data if x["status"] == "below")

    return (
        f"You spend above average in {above} categories "
        f"and below average in {below} categories."
    )