from sklearn.impute import SimpleImputer
import logging 
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)

datapath = "data/clean_preprocessed.csv"

# loop through the categorical data and fill with most frequent
def filling_categorical(data:pd.DataFrame):
    data = pd.read_csv(data)
    categorical_data = data.select_dtypes(exclude='number').columns.tolist()
    for x in categorical_data:
        if data[x].isna().sum() > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            data[x] = imputer.fit_transform(data[[x]]).ravel()
    return data


# loop through the categorical data and fill with most frequent
def filling_categoricals(data:pd.DataFrame):
    numerical_data = data.select_dtypes(include='number').columns.tolist()
    for x in numerical_data:
        if data[x].isna().sum() > 0:
            imputer = SimpleImputer(strategy='mean')
            data[x] = imputer.fit_transform(data[[x]]).ravel()
    return data



# make sure these columns are numeric,creating uid
def uid_policy(data:pd.DataFrame):
    for col in ["TransactionDT", "D1", "card1", "addr1"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # TransactionDT is in seconds, so convert to days
    day = data["TransactionDT"] / (24 * 60 * 60)

    # day the account started: constant for the same client
    data["account_start_day"] = np.floor(day - data["D1"])

    # glue the clues together into one client label
    data["uid"] = (
        data["card1"].astype(str) + "_"
        + data["addr1"].astype(str) + "_"
        + data["account_start_day"].astype(str)
    )

    has_nan = data["uid"].str.contains("nan")
    data.loc[has_nan, "uid"] = "unknown"
    return data 


#create zscore
def columns_zscore(data:pd.DataFrame):
    #create the zscore
    data['zscore'] = (data['TransactionAmt']- data['TransactionAmt'].mean()
                 )/data['TransactionAmt'].std()
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


    #proxixy
def columns_proxy(data:pd.DataFrame):
        # how many different customers share this proxy (s
    data['proxy_uid_count'] = data.groupby('device_proxy')['uid'].transform('nunique')

        # how many different proxies this customer uses
    data['uid_proxy_count'] = data.groupby('uid')['device_proxy'].transform('nunique')

        # how many transactions this proxy has
    data['proxy_txn_count'] = data.groupby('device_proxy')['TransactionAmt'].transform('count')
    return data

#transaction velocity
def transaction_velocity(data:pd.DataFrame):
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
    #seperating the year, month and day from the dt date 
    data = data.assign(
        year=data['dt'].dt.year,
        month=data['dt'].dt.month,
        day=data['dt'].dt.day,
        )
    return data



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = filling_categorical(datapath)          
    data = filling_categoricals(data)          
    data = uid_policy(data) 
    data = columns_zscore(data) 
    data = columns_proxy(data)
    data = transaction_velocity(data)
    data.to_csv("data/new_clean_preprocessed.csv", index=False) 

    print(data.sample(5))       

