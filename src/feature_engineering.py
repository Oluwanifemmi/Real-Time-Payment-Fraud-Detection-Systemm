"""
get_historical.py

1. Reads the offline store (parquet) directly with pandas.
2. Pulls the same rows through Feast's get_historical_features (point-in-time join).
3. Prints 5 samples from each.

Run from the project root:
    python feature_store/get_historical.py
"""

import sys
from pathlib import Path

import pandas as pd
from feast import FeatureStore

HERE = Path(__file__).resolve().parent          # .../paymentfraud/src
PROJECT_ROOT = HERE.parent                      # .../paymentfraud
REPO_DIR = PROJECT_ROOT / "feature_store" / "features_repo"
PARQUET_PATH = REPO_DIR / "fraud_transaction.parquet"

sys.path.insert(0, str(REPO_DIR))
from card_features import fraud_features  # noqa: E402

FEATURE_COLUMNS = [f.name for f in fraud_features.features]
FEATURE_REFS = [f"{fraud_features.name}:{c}" for c in FEATURE_COLUMNS]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def main() -> None:
    # 1. read the parquet directly
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Parquet: {len(df):,} rows, {df['uid'].nunique():,} uids, {df.shape[1]} columns\n")
    print("5 samples straight from the parquet:")
    print(df.sample(5, random_state=42).to_string(index=False), "\n")

    # 2. historical features through Feast
    # the entity dataframe says WHICH uid at WHICH time we want features for
    entity_df = (
        df[["uid", "event_timestamp"]]
        .sample(5, random_state=42)
        .assign(event_timestamp=lambda x: pd.to_datetime(x["event_timestamp"], utc=True))
        .reset_index(drop=True)
    )

    store = FeatureStore(repo_path=str(REPO_DIR))
    hist = store.get_historical_features(
        entity_df=entity_df,
        features=FEATURE_REFS,
    ).to_df()

    print("5 samples from Feast get_historical_features:")
    print(hist.to_string(index=False))


if __name__ == "__main__":
    main()