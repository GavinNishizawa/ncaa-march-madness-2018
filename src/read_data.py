'''
Read in the data from csv files
'''

import pandas as pd
import numpy as np
import os
import pickle

def load_csv(filename):
    pickle_fn = filename+".pkl"

    # load data from pickle file if it exists
    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))
    # otherwise load from csv
    else:
        data = pd.read_csv(filename)
        pickle.dump(data, open(pickle_fn, 'wb'))
        return data


def main():
    print("Test loading csv")
    data = load_csv("data/NCAATourneySeeds.csv")
    print(data)

if __name__=="__main__":
    main()

