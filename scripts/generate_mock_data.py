"""Generate synthetic credit-risk data for local testing and CI.

Mimics the structure of Kaggle Home Credit Default Risk:
- ~30K rows, 20 features (mix of numeric and categorical)
- 8% default rate (class imbalance)
- realistic correlation patterns
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, RANDOM_STATE


def generate_application_train(n_rows: int = 30000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic application_train.csv."""
    rng = np.random.RandomState(seed)

    X, y = make_classification(
        n_samples=n_rows,
        n_features=15,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=[0.92, 0.08],
        flip_y=0.02,
        random_state=seed,
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(15)])
    df["TARGET"] = y

    # Add realistic credit columns
    df["SK_ID_CURR"] = rng.randint(100000, 999999, n_rows)
    df["AMT_INCOME_TOTAL"] = rng.lognormal(12, 0.8, n_rows).astype(int)
    df["AMT_CREDIT"] = (df["AMT_INCOME_TOTAL"] * rng.uniform(0.5, 5.0, n_rows)).astype(int)
    df["AMT_ANNUITY"] = (df["AMT_CREDIT"] * rng.uniform(0.03, 0.08, n_rows)).astype(int)
    df["DAYS_BIRTH"] = -rng.randint(6500, 25500, n_rows)
    df["DAYS_EMPLOYED"] = -rng.randint(0, 18000, n_rows)
    df["DAYS_REGISTRATION"] = -rng.randint(500, 9000, n_rows)
    df["DAYS_ID_PUBLISH"] = -rng.randint(500, 8000, n_rows)
    df["OWN_CAR_AGE"] = rng.choice([np.nan] + list(range(0, 65)), n_rows, p=[0.3] + [0.7 / 65] * 65)
    df["CNT_CHILDREN"] = rng.poisson(0.5, n_rows)
    df["CNT_FAM_MEMBERS"] = df["CNT_CHILDREN"] + rng.poisson(1.5, n_rows)
    df["REGION_RATING_CLIENT"] = rng.randint(1, 4, n_rows)
    df["REGION_RATING_CLIENT_W_CITY"] = rng.randint(1, 4, n_rows)
    df["EXT_SOURCE_1"] = rng.beta(2, 5, n_rows)
    df["EXT_SOURCE_2"] = rng.beta(2, 5, n_rows)
    df["EXT_SOURCE_3"] = rng.beta(2, 5, n_rows)

    # Categoricals
    df["NAME_CONTRACT_TYPE"] = rng.choice(["Cash loans", "Revolving loans"], n_rows, p=[0.9, 0.1])
    df["CODE_GENDER"] = rng.choice(["M", "F", "XNA"], n_rows, p=[0.45, 0.54, 0.01])
    df["FLAG_OWN_CAR"] = rng.choice(["Y", "N"], n_rows, p=[0.35, 0.65])
    df["FLAG_OWN_REALTY"] = rng.choice(["Y", "N"], n_rows, p=[0.7, 0.3])
    df["NAME_TYPE_SUITE"] = rng.choice(
        ["Unaccompanied", "Family", "Spouse, partner", "Children", "Other_B", "Other_A", np.nan],
        n_rows,
        p=[0.6, 0.15, 0.1, 0.08, 0.03, 0.02, 0.02],
    )
    df["NAME_INCOME_TYPE"] = rng.choice(
        ["Working", "State servant", "Commercial associate", "Pensioner", "Unemployed"],
        n_rows,
        p=[0.55, 0.08, 0.2, 0.15, 0.02],
    )
    df["NAME_EDUCATION_TYPE"] = rng.choice(
        ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"],
        n_rows,
        p=[0.7, 0.2, 0.06, 0.03, 0.01],
    )
    df["NAME_FAMILY_STATUS"] = rng.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        n_rows,
        p=[0.6, 0.2, 0.1, 0.07, 0.03],
    )
    df["NAME_HOUSING_TYPE"] = rng.choice(
        ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"],
        n_rows,
        p=[0.9, 0.05, 0.02, 0.02, 0.005, 0.005],
    )
    df["OCCUPATION_TYPE"] = rng.choice(
        [
            "Laborers", "Sales staff", "Core staff", "Managers", "Drivers",
            "High skill tech staff", "Accountants", "Medicine staff", "Security staff",
            "Cooking staff", "Cleaning staff", "Private service staff", "Low-skill Laborers",
            "Waiters/barmen staff", "Secretaries", "Realty agents", "HR staff", "IT staff",
            np.nan,
        ],
        n_rows,
        p=[0.15, 0.10, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.04],
    )
    df["WEEKDAY_APPR_PROCESS_START"] = rng.choice(
        ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"],
        n_rows,
    )
    df["ORGANIZATION_TYPE"] = rng.choice(
        [
            "Business Entity Type 3", "Business Entity Type 2", "Self-employed",
            "Government", "Medicine", "Education", "Other", np.nan,
        ],
        n_rows,
        p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.05, 0.02],
    )

    # Introduce some missing values realistically
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE"]:
        mask = rng.random(n_rows) < 0.2
        df.loc[mask, col] = np.nan

    # Shuffle columns to match realistic order
    cols = ["SK_ID_CURR", "TARGET"] + [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]
    return df[cols]


def generate_application_test(n_rows: int = 5000, seed: int = RANDOM_STATE + 1) -> pd.DataFrame:
    """Generate synthetic application_test.csv (no TARGET)."""
    df = generate_application_train(n_rows=n_rows, seed=seed)
    return df.drop(columns=["TARGET"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-rows", type=int, default=30000)
    parser.add_argument("--test-rows", type=int, default=5000)
    parser.add_argument("--force", action="store_true", help="Overwrite existing real data")
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = RAW_DATA_DIR / "application_train.csv"
    test_path = RAW_DATA_DIR / "application_test.csv"

    if train_path.exists() and test_path.exists() and not args.force:
        print(f"Real data already exists at {train_path} and {test_path}")
        print("Skipping mock data generation. Use --force to overwrite.")
        return

    train_df = generate_application_train(args.train_rows)
    test_df = generate_application_test(args.test_rows)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Generated: {train_path} ({len(train_df):,} rows, {train_df['TARGET'].mean()*100:.2f}% default rate)")
    print(f"Generated: {test_path} ({len(test_df):,} rows)")


if __name__ == "__main__":
    main()
