import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

data_csv = pd.read_csv("data/clean_preprocessed.csv")


#create zscore
def columns_zscore(data: pd.DataFrame):
    data['TransactionAmt'] = pd.to_numeric(data['TransactionAmt'], errors='coerce')
    #create the zscore
    data['zscore'] = (data['TransactionAmt'] - data['TransactionAmt'].mean()
                      ) / data['TransactionAmt'].std()
    # zscore per amount
    g_uid = data.groupby('uid')['TransactionAmt']
    data['zscore_uid'] = (data['TransactionAmt'] - g_uid.transform('mean')) / g_uid.transform('std')
    data['zscore_uid'] = data['zscore_uid'].replace([float('inf'), float('-inf')], 0).fillna(0)
    # zscore per ProductCD
    g_prod = data.groupby('ProductCD')['TransactionAmt']
    data['zscore_productcd'] = (data['TransactionAmt'] - g_prod.transform('mean')) / g_prod.transform('std')
    data['zscore_productcd'] = data['zscore_productcd'].replace([float('inf'), float('-inf')], 0).fillna(0)
    #filling in missing data for device proxy
    proxy_cols = ['DeviceType', 'card4', 'card6', 'P_emaildomain']
    data['device_proxy'] = data[proxy_cols].fillna('missing').astype(str).agg('_'.join, axis=1)
    return data


#proxy
def columns_proxy(data: pd.DataFrame):
    # how many different customers share this proxy
    data['proxy_uid_count'] = data.groupby('device_proxy')['uid'].transform('nunique')

    # how many different proxies this customer uses
    data['uid_proxy_count'] = data.groupby('uid')['device_proxy'].transform('nunique')

    # how many transactions this proxy has
    data['proxy_txn_count'] = data.groupby('device_proxy')['TransactionAmt'].transform('count')
    return data


#transaction velocity
def transaction_velocity(data: pd.DataFrame):
    data = data.sort_values(['uid', 'TransactionDT']).reset_index(drop=True)
    data['dt'] = pd.to_datetime(data['TransactionDT'], unit='s')  # skip if already datetime

    new_cols = {}
    for window in ['1h', '24h']:
        r = data.groupby('uid').rolling(window, on='dt', closed='left')['TransactionAmt']
        cnt = r.count().to_numpy()
        sm = r.sum().to_numpy()
        assert len(cnt) == len(data), f"length mismatch: {len(cnt)} vs {len(data)}"
        new_cols[f'txn_count_{window}'] = np.nan_to_num(cnt, nan=0.0)
        new_cols[f'txn_amt_sum_{window}'] = np.nan_to_num(sm, nan=0.0)

    data = data.assign(**new_cols)
    #separating the year, month and day from the dt date
    data = data.assign(
        year=data['dt'].dt.year,
        month=data['dt'].dt.month,
        day=data['dt'].dt.day,
    )
    return data


# dtypes must match the schema in card_features.py
INT_COLS = ["txn_count_1h", "txn_count_24h", "year", "month", "day",
            "proxy_uid_count", "uid_proxy_count", "proxy_txn_count"]
FLOAT_COLS = ["txn_amt_sum_1h", "txn_amt_sum_24h", "zscore", "zscore_uid",
              "zscore_productcd", "account_start_day", "TransactionAmt"]


def build_parquet(data: pd.DataFrame):
    data = data[["txn_count_1h", "txn_amt_sum_1h", "txn_count_24h", "txn_amt_sum_24h",
                 "year", "month", "day", "uid", "zscore", "zscore_uid", "zscore_productcd",
                 "device_proxy", "proxy_uid_count", "uid_proxy_count", "proxy_txn_count",
                 "DeviceType", "account_start_day", "TransactionAmt", "dt"]].copy()

    # Feast needs a timestamp column called event_timestamp
    data = data.rename(columns={"dt": "event_timestamp"})
    data["event_timestamp"] = pd.to_datetime(data["event_timestamp"], utc=True)

    # Feast needs a string entity key
    data["uid"] = data["uid"].astype(str)

    # strings: no NaNs allowed
    data["device_proxy"] = data["device_proxy"].astype(str)
    data["DeviceType"] = data["DeviceType"].fillna("unknown").astype(str)

    # numbers: match Int64 / Float32 in card_features.py
    data[INT_COLS] = data[INT_COLS].astype("int64")
    data[FLOAT_COLS] = data[FLOAT_COLS].astype("float32")
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    data = columns_zscore(data_csv)
    data = columns_proxy(data)
    data = transaction_velocity(data)
    data = build_parquet(data)
    logger.info(f"Writing {len(data):,} rows\n{data.dtypes}")
    data.to_parquet("feature_store/features_repo/fraud_transaction.parquet", index=False)