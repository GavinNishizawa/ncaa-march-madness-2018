"""
Read in the data from csv files
"""
import pandas as pd
import os
import glob
from save_data import save_object, load_object


def load_csv(filename):

    # load data from pickle file if it exists
    obj = load_object(filename)
    if obj != None:
        return obj

    # otherwise load from csv
    else:
        data = pd.read_csv(filename, encoding="latin_1")
        save_object(filename, data)
        return data


def load_data():
    pickle_fn = "data/loaded_data"

    data = load_object(pickle_fn)
    # load data from pickle file if it exists
    if data != None:
        return data

    # otherwise load from csv
    else:
        data = {}

        # load all csv files in data directory
        for f in glob.glob(os.path.join("data", "*.csv")):
            # key based on their filename
            f_key = os.path.basename(f).split('.')[0]

            print("Loading:", f_key)
            data[f_key] = load_csv(f)

        save_object(pickle_fn, data)
        return data


def test_load_csv():
    print("Test loading csv")
    data = load_csv("data/NCAATourneySeeds.csv")
    print(data)


def main():
    data = load_data()
    print("Available DataSet Keys: ")
    for key in data.keys():
        print("\t"+key)


if __name__ == "__main__":
    main()

