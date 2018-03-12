"""
Read in the data from csv files
"""
import pandas as pd
import os
import glob
import pickle


def load_csv(filename):
    pickle_fn = filename + ".pkl"

    # load data from pickle file if it exists
    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    # otherwise load from csv
    else:
        data = pd.read_csv(filename, encoding="latin_1")
        pickle.dump(data, open(pickle_fn, 'wb'))
        return data


def load_data():
    pickle_fn = "data/loaded_data.pkl"

    # load data from pickle file if it exists
    if os.path.isfile(pickle_fn):
        return pickle.load(open(pickle_fn, 'rb'))

    # otherwise load from csv
    else:
        df_dict = {}

        # load all csv files in data directory
        for f in glob.glob(os.path.join("data", "*.csv")):
            # key based on their filename
            f_key = os.path.basename(f).split('.')[0]

            print("Loading:", f_key)
            df_dict[f_key] = load_csv(f)

        pickle.dump(df_dict, open(pickle_fn, 'wb'))
        return df_dict


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

