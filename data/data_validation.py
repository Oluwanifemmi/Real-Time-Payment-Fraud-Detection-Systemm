import pandas as pd



data = "data/payment.csv"
train_identity_PATH = "data/train_identity.csv"
train_transaction_path = "data/train_transaction.csv"

#mapping the two raw data 
def mapping_raw_data(identity, transaction):
    identity = pd.read_csv(identity)
    transaction = pd.read_csv(transaction)
    new_data =transaction.merge(identity, how ='left', on ='TransactionID')
    return new_data.sample(10)


mapping_raw_data(train_identity_PATH,train_transaction_path)