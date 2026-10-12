import pandas as pd
import logging


logger = logging.getLogger(__name__)

datapath = 'data_validation/expectation/processed.csv'

def data_process(data:str):
    try:
        data = pd.read_csv(data)
        logging.info("importing of the processed data")
    except FileNotFoundError:
        logger.exception("couldnot read data file")
        raise
        
    return data



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = data_process(datapath)
    print(result.sample(5))

