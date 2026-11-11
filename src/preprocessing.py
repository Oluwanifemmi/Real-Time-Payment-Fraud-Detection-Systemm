import pandas as pd
import logging
import numpy as np
from sklearn.impute import SimpleImputer


logger = logging.getLogger(__name__)

datapath = 'data_validation/expectation/processed.csv'

def data_process(data:str):
    try:
        data = pd.read_csv(data)
        logging.info("importing of the processed data")
    except FileNotFoundError:
        logger.exception("couldnot read data file")
        raise
    logger.info("completed data_process")
    return data

#data preprocessing to  remove the columns that have mostly 0 and 1, getting 95% of the data, set threshold to 95
def processing_binary_columns(data:pd.DataFrame):
    threshold_pct = 0.95

    cols_to_check = []
    
    for c in data.select_dtypes(include="number").columns:
        if c not in ("isFraud", "TransactionID"):
            cols_to_check.append(c)

    low_variance_cols = []

    for col in cols_to_check:
        value_counts = data[col].value_counts(normalize=True, dropna=False)
        pct_01 = value_counts.reindex([0, 1], fill_value=0).sum()
        if pct_01 > threshold_pct:
         low_variance_cols.append(col)
    data = data.drop(columns=low_variance_cols)
    logger.info(f"the low variance columns include{low_variance_cols}")
    return data


#function to remove columns with more than 200,000 missing rowa
def missing_columns(data:pd.DataFrame):
    threshold = 200000
    keep_cols = ["DeviceType", "card4", "card6"]

    missing_counts = data.isna().sum()
    cols_to_drop =[]
    for col, count in missing_counts.items():
        if count >= threshold:
            if col not in keep_cols:
                cols_to_drop.append(col)
    data = data.drop(columns=cols_to_drop)
    logger.info(f"the columns dropped include {cols_to_drop}")
    return data


#removing the columns that are higly correlated
def process_corelated(data: pd.DataFrame):
    corr_threshold = 0.95
    exclude = {"isFraud", "TransactionID"}

    num_cols = []
    for c in data.select_dtypes(include="number").columns:
        if c not in exclude:
            num_cols.append(c)

    corr_matrix = data[num_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    corr_cols_to_drop = []
    for col in upper.columns:
        if (upper[col] >= corr_threshold).any():
            corr_cols_to_drop.append(col)

    logger.info("Dropping %d highly correlated columns", len(corr_cols_to_drop))
    data = data.drop(columns=corr_cols_to_drop)
    return data



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
        + data["account_start_day"].astype(str))
    has_nan = data["uid"].str.contains("nan")
    data.loc[has_nan, "uid"] = "unknown"
    return data 



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    data = data_process(datapath)          
    data = missing_columns(data)          
    data = processing_binary_columns(data) 
    data = process_corelated(data) 
    data = filling_categorical(datapath)          
    data = filling_categoricals(data)          
    data = uid_policy(data)


    data.to_csv("data/clean_preprocessed.csv", index=False)        

    print(data.columns)
    print(data.sample(5))

