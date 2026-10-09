import pandas as pd
import numpy as np



#mapping the two raw data 
def mapping_raw_data(x, y):
    identity = pd.read_csv("data/train_identity.csv")
    transaction = pd.read_csv("data/train_transaction.csv")
    main_data = identity.join
    return mapping_raw_data


mapping_raw_data()