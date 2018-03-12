'''
Read in the data from csv files
'''

import pandas as pd
import numpy as np

def load_csv(filename):
    return pd.read_csv(filename)


def main():
    print("Test loading csv")
    data = load_csv("data/NCAATourneySeeds.csv")
    print(data)

if __name__=="__main__":
    main()

