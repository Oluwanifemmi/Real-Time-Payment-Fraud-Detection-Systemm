import pandas as pd
import logging


logger = logging.getLogger(__name__)

train_identity_path = "data/train_identity.csv"
train_transaction_path = "data/train_transaction.csv"


#mapping the two raw data 
def mapping_raw_data(identity :str, transaction:str):
    try:
        identity_data = pd.read_csv(identity)
        transaction_data = pd.read_csv(transaction)
        new_data =transaction_data.merge(identity_data, how ='left', on ='TransactionID')
    except FileNotFoundError :
        logger.exception("Could not read data files: %s, %s", identity, transaction)
        raise
    logger.info(f"merge raw data from both {identity} and {transaction}")
    return new_data


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     result = mapping_raw_data(train_identity_path, train_transaction_path)
#     print(result.columns)
