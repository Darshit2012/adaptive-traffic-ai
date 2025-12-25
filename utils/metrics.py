"""Metrics utilities for comparing controllers."""
import pandas as pd


def summarize_metrics(df: pd.DataFrame):
    metrics = {}
    metrics["avg_wait_proxy"] = round(df["avg_wait_proxy"].mean(), 3)
    metrics["throughput"] = int(df["throughput"].sum())
    metrics["stops"] = int(df["stops"].sum())
    metrics["emergencies_handled"] = int(df["emergency"].sum())
    return metrics


def comparison_table(results: dict) -> pd.DataFrame:
    """Build comparison table from {controller: metrics dict}."""
    return pd.DataFrame(results).T
