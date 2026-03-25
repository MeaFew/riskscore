"""Aggregate auxiliary tables into features for the main application table.

Home Credit Default Risk dataset contains multiple auxiliary tables:
- bureau: external credit bureau records
- bureau_balance: monthly status of bureau credits
- previous_application: previous loan applications
- POS_CASH_balance: monthly POS/cash loan balances
- credit_card_balance: monthly credit card balances
- installments_payments: installment payment history

This script extracts aggregate features from each table and merges them
into the main application_train table by SK_ID_CURR.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def aggregate_bureau(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau.csv features per SK_ID_CURR."""
    grp = df.groupby("SK_ID_CURR")
    agg = pd.DataFrame()
    agg["bureau_count"] = grp.size()
    agg["bureau_active_count"] = grp.apply(lambda x: (x["CREDIT_ACTIVE"] == "Active").sum())
    agg["bureau_active_ratio"] = agg["bureau_active_count"] / agg["bureau_count"]
    agg["bureau_avg_days_credit"] = grp["DAYS_CREDIT"].mean()
    agg["bureau_max_days_credit"] = grp["DAYS_CREDIT"].max()
    agg["bureau_avg_credit_sum"] = grp["AMT_CREDIT_SUM"].mean()
    agg["bureau_sum_credit_sum"] = grp["AMT_CREDIT_SUM"].sum()
    agg["bureau_avg_credit_debt"] = grp["AMT_CREDIT_SUM_DEBT"].mean()
    agg["bureau_avg_credit_overdue"] = grp["AMT_CREDIT_SUM_OVERDUE"].mean()
    agg["bureau_max_credit_overdue"] = grp["AMT_CREDIT_SUM_OVERDUE"].max()
    agg["bureau_avg_day_overdue"] = grp["CREDIT_DAY_OVERDUE"].mean()
    agg["bureau_max_day_overdue"] = grp["CREDIT_DAY_OVERDUE"].max()
    agg["bureau_avg_annuity"] = grp["AMT_ANNUITY"].mean()
    agg["bureau_sum_annuity"] = grp["AMT_ANNUITY"].sum()
    agg["bureau_avg_prolong"] = grp["CNT_CREDIT_PROLONG"].mean()
    return agg.reset_index()


def aggregate_bureau_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bureau_balance.csv features per SK_ID_BUREAU, then merge with bureau."""
    # Status distribution per bureau record
    status_counts = df.groupby("SK_ID_BUREAU")["STATUS"].value_counts().unstack(fill_value=0)
    status_counts["bureau_bal_total_months"] = status_counts.sum(axis=1)
    # C = closed, X = unknown, 0 = no DPD, 1-5 = DPD buckets
    status_counts["bureau_bal_closed_ratio"] = status_counts.get("C", 0) / status_counts["bureau_bal_total_months"]
    status_counts["bureau_bal_no_dpd_ratio"] = status_counts.get("0", 0) / status_counts["bureau_bal_total_months"]
    # Any overdue (status 1-5)
    overdue_cols = [c for c in status_counts.columns if c in "12345"]
    status_counts["bureau_bal_overdue_ratio"] = status_counts[overdue_cols].sum(axis=1) / status_counts["bureau_bal_total_months"]
    status_counts = status_counts.reset_index()

    # Merge with bureau to get SK_ID_CURR
    bureau = pd.read_csv(RAW_DATA_DIR / "bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
    merged = status_counts.merge(bureau, on="SK_ID_BUREAU", how="left")
    # Aggregate per SK_ID_CURR
    grp = merged.groupby("SK_ID_CURR")
    agg = pd.DataFrame()
    agg["bureau_bal_avg_months"] = grp["bureau_bal_total_months"].mean()
    agg["bureau_bal_avg_closed_ratio"] = grp["bureau_bal_closed_ratio"].mean()
    agg["bureau_bal_avg_no_dpd_ratio"] = grp["bureau_bal_no_dpd_ratio"].mean()
    agg["bureau_bal_avg_overdue_ratio"] = grp["bureau_bal_overdue_ratio"].mean()
    return agg.reset_index()


def aggregate_previous_application(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous_application.csv features per SK_ID_CURR."""
    grp = df.groupby("SK_ID_CURR")
    agg = pd.DataFrame()
    agg["prev_app_count"] = grp.size()
    agg["prev_app_approved_count"] = grp.apply(lambda x: (x["NAME_CONTRACT_STATUS"] == "Approved").sum())
    agg["prev_app_approved_ratio"] = agg["prev_app_approved_count"] / agg["prev_app_count"]
    agg["prev_app_avg_annuity"] = grp["AMT_ANNUITY"].mean()
    agg["prev_app_sum_annuity"] = grp["AMT_ANNUITY"].sum()
    agg["prev_app_avg_credit"] = grp["AMT_CREDIT"].mean()
    agg["prev_app_sum_credit"] = grp["AMT_CREDIT"].sum()
    agg["prev_app_avg_application"] = grp["AMT_APPLICATION"].mean()
    agg["prev_app_avg_down_payment"] = grp["AMT_DOWN_PAYMENT"].mean()
    agg["prev_app_avg_goods_price"] = grp["AMT_GOODS_PRICE"].mean()
    agg["prev_app_avg_rate_down_payment"] = grp["RATE_DOWN_PAYMENT"].mean()
    agg["prev_app_avg_decision_days"] = grp["DAYS_DECISION"].mean()
    agg["prev_app_avg_cnt_payment"] = grp["CNT_PAYMENT"].mean()
    agg["prev_app_avg_days_first_due"] = grp["DAYS_FIRST_DUE"].mean()
    agg["prev_app_avg_days_last_due"] = grp["DAYS_LAST_DUE"].mean()
    return agg.reset_index()


def aggregate_pos_cash(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance.csv features per SK_ID_CURR."""
    grp = df.groupby("SK_ID_CURR")
    agg = pd.DataFrame()
    agg["pos_cash_count"] = grp.size()
    agg["pos_cash_avg_dpd"] = grp["SK_DPD"].mean()
    agg["pos_cash_max_dpd"] = grp["SK_DPD"].max()
    agg["pos_cash_avg_dpd_def"] = grp["SK_DPD_DEF"].mean()
    agg["pos_cash_avg_instalment"] = grp["CNT_INSTALMENT"].mean()
    agg["pos_cash_avg_instalment_future"] = grp["CNT_INSTALMENT_FUTURE"].mean()
    agg["pos_cash_active_ratio"] = grp.apply(lambda x: (x["NAME_CONTRACT_STATUS"] == "Active").sum()) / agg["pos_cash_count"]
    return agg.reset_index()


def aggregate_credit_card(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance.csv features per SK_ID_CURR."""
    grp = df.groupby("SK_ID_CURR")
    agg = pd.DataFrame()
    agg["cc_count"] = grp.size()
    agg["cc_avg_balance"] = grp["AMT_BALANCE"].mean()
    agg["cc_avg_credit_limit"] = grp["AMT_CREDIT_LIMIT_ACTUAL"].mean()
    agg["cc_avg_drawings"] = grp["AMT_DRAWINGS_CURRENT"].mean()
    agg["cc_avg_drawings_atm"] = grp["AMT_DRAWINGS_ATM_CURRENT"].mean()
    agg["cc_avg_payment"] = grp["AMT_PAYMENT_CURRENT"].mean()
    agg["cc_avg_payment_total"] = grp["AMT_PAYMENT_TOTAL_CURRENT"].mean()
    agg["cc_avg_min_regularity"] = grp["AMT_INST_MIN_REGULARITY"].mean()
    agg["cc_avg_receivable"] = grp["AMT_TOTAL_RECEIVABLE"].mean()
    agg["cc_avg_cnt_drawings"] = grp["CNT_DRAWINGS_CURRENT"].mean()
    agg["cc_avg_cnt_drawings_atm"] = grp["CNT_DRAWINGS_ATM_CURRENT"].mean()
    agg["cc_avg_dpd"] = grp["SK_DPD"].mean()
    agg["cc_max_dpd"] = grp["SK_DPD"].max()
    agg["cc_active_ratio"] = grp.apply(lambda x: (x["NAME_CONTRACT_STATUS"] == "Active").sum()) / agg["cc_count"]
    # Credit utilization
    agg["cc_avg_utilization"] = grp.apply(
        lambda x: (x["AMT_BALANCE"] / (x["AMT_CREDIT_LIMIT_ACTUAL"] + 1)).mean()
    )
    return agg.reset_index()


def aggregate_installments(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments.csv features per SK_ID_CURR."""
    # Payment ratio
    df["payment_ratio"] = df["AMT_PAYMENT"] / (df["AMT_INSTALMENT"] + 1)
    df["days_diff"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    grp = df.groupby("SK_ID_CURR")
    agg = pd.DataFrame()
    agg["inst_count"] = grp.size()
    agg["inst_avg_payment_ratio"] = grp["payment_ratio"].mean()
    agg["inst_min_payment_ratio"] = grp["payment_ratio"].min()
    agg["inst_avg_days_diff"] = grp["days_diff"].mean()
    agg["inst_max_days_diff"] = grp["days_diff"].max()
    agg["inst_late_count"] = grp.apply(lambda x: (x["days_diff"] > 0).sum())
    agg["inst_late_ratio"] = agg["inst_late_count"] / agg["inst_count"]
    agg["inst_avg_instalment"] = grp["AMT_INSTALMENT"].mean()
    agg["inst_sum_instalment"] = grp["AMT_INSTALMENT"].sum()
    agg["inst_avg_payment"] = grp["AMT_PAYMENT"].mean()
    agg["inst_sum_payment"] = grp["AMT_PAYMENT"].sum()
    return agg.reset_index()


def main():
    print("Loading auxiliary tables ...")
    bureau = pd.read_csv(RAW_DATA_DIR / "bureau.csv")
    bureau_balance = pd.read_csv(RAW_DATA_DIR / "bureau_balance.csv")
    prev_app = pd.read_csv(RAW_DATA_DIR / "previous_application.csv")
    pos_cash = pd.read_csv(RAW_DATA_DIR / "POS_CASH_balance.csv")
    cc_balance = pd.read_csv(RAW_DATA_DIR / "credit_card_balance.csv")
    installments = pd.read_csv(RAW_DATA_DIR / "installments_payments.csv")

    features = []

    print("Aggregating bureau ...")
    features.append(aggregate_bureau(bureau))

    print("Aggregating bureau_balance ...")
    features.append(aggregate_bureau_balance(bureau_balance))

    print("Aggregating previous_application ...")
    features.append(aggregate_previous_application(prev_app))

    print("Aggregating POS_CASH_balance ...")
    features.append(aggregate_pos_cash(pos_cash))

    print("Aggregating credit_card_balance ...")
    features.append(aggregate_credit_card(cc_balance))

    print("Aggregating installments_payments ...")
    features.append(aggregate_installments(installments))

    # Merge all on SK_ID_CURR
    print("Merging all auxiliary features ...")
    merged = features[0]
    for feat in features[1:]:
        merged = merged.merge(feat, on="SK_ID_CURR", how="outer")

    # Fill NAs with 0 (no record = 0 for counts/ratios)
    merged = merged.fillna(0)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "auxiliary_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged):,} rows × {len(merged.columns)} cols to {out_path}")


if __name__ == "__main__":
    main()
