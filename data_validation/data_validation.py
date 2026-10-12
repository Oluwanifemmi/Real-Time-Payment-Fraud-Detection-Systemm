from data.data_loader import mapping_raw_data

import json
import os
import logging

import pandas as pd
from great_expectations.dataset import PandasDataset

logger = logging.getLogger(__name__)

#direct part to add clean data and expectation folder
EXPECTATION_DIR = "data_validation/expectation"

#raw data folder path
train_identity = "data/train_identity.csv"
train_transaction = "data/train_transaction.csv"

#checking the files in the folder
os.makedirs(EXPECTATION_DIR, exist_ok=True)

productcd_set = ['W','C','R','H','S']
card4_set = ['visa','mastercard','american express','discover']
card6_set = ['debit','credit','debit or credit','charge card']
device_set = ['desktop','mobile']

#performing data validation check on just 10columns from the data 
"""'TransactionID,'TransactionDT',
 'TransactionAmt', 'DeviceType', 'card4',
   'card6','is_fraud"""



def data_validation(data:str):
    gx_df = PandasDataset(data)
    # Record of the checks
    expectation_suite = {
                "TransactionID": {"unique": True, "not_null": True},
                "ProductCD" : {"in_set":productcd_set},
                "card4" : {"in_set": card4_set},
                "TransactionDT": {"not_null": True, "min": 1},
                "card6" : {"in_set" : card6_set},
                "DeviceType": {"in_set": device_set},
                "TransactionAmt": {"min": 0, "max": 32000},
                "isFraud": {"min": 0, "max": 1, "not_null": True},}
    logger.info(f"completed building the {expectation_suite}")

    with open(f"{EXPECTATION_DIR}/expectation_suite.json", "w") as f:
        json.dump(expectation_suite, f, indent=4)
        logger.info("expectation suite saved")

        checks = {}

        checks["TransactionID"] = gx_df.expect_column_values_to_be_unique("TransactionID").success
        checks["TransactionDT"] = gx_df.expect_column_values_to_not_be_null("TransactionDT").success
        checks["TransactionAmt"] = gx_df.expect_column_values_to_be_between("TransactionAmt", min_value=0, max_value=32000).success
        checks["ProductCD"] = gx_df.expect_column_values_to_be_in_set("ProductCD", productcd_set).success
        checks["card6"] = gx_df.expect_column_values_to_be_in_set("card6", card6_set).success
        checks["card4"] = gx_df.expect_column_values_to_be_in_set("card4", card4_set).success
        checks["devicetype"] = gx_df.expect_column_values_to_be_in_set("DeviceType", device_set).success
        checks["isFraud"] = gx_df.expect_column_values_to_be_between("isFraud", min_value=0, max_value=1).success
        checks["isFraud_notnull"] = gx_df.expect_column_values_to_not_be_null("isFraud").success
        logger.info("expectation suitechecks completed")

    # Validation summary
    with open(f"{EXPECTATION_DIR}/validation_results.txt", "w") as f:
        for check, status in checks.items():
            f.write(f"[{'passed' if status else 'failed'}] {check}\n")
    logger.info("validation results saved to data_validation folder")

    def not_in_set(col, allowed):
        return data[col].notna() & ~data[col].isin(allowed)

    #validationchecks
    invalid_mask = (
        data["TransactionID"].isnull()
        | data["TransactionID"].duplicated(keep=False)
        | data["TransactionDT"].isnull()
        | not_in_set("ProductCD", productcd_set)
        | not_in_set("card4", card4_set)
        | not_in_set("card6", card6_set)
        | not_in_set("DeviceType", device_set)
        | ~data["TransactionAmt"].between(0, 32000)
        | ~data["isFraud"].isin([0, 1])
    )

    invalid_df = data[invalid_mask]
    clean_df = data[~invalid_mask]
    invalid_df.to_csv(f"{EXPECTATION_DIR}/failed.csv", index=False)
    clean = clean_df.to_csv(f"{EXPECTATION_DIR}/processed.csv", index=False)
    logger.info("Valid rows: %d, invalid rows: %d", len(clean_df), len(invalid_df))
    return clean_df, invalid_df
    



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw = mapping_raw_data(train_identity,train_transaction)
    clean, invalid = data_validation(raw)
    print(clean.shape, invalid.shape)
