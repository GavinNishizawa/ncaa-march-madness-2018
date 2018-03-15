"""
Read in the results data from csv files
"""
import pandas as pd
import os
import glob


def load_csv(filename):
    # load from csv
    data = pd.read_csv(filename, encoding="latin_1")
    return data


def load_data():
    data = {}

    # load all csv files in models directory
    for f in glob.glob(os.path.join("src","models", \
            "trained_*_results.csv")):
        # key based on their filename
        f_key = os.path.basename(f).split('.')[0]

        data[f_key] = load_csv(f)

    return data


def get_averages(data):
    d_avg = {}
    for key in data.keys():
        d_avg[key] = [
                data[key].ix[:,0].mean(),
                data[key].ix[:,1].mean(),
                data[key].ix[:,2].mean()]
    return d_avg


def main():
    data = load_data()
    print("Available DataSet Keys: ")
    for key in data.keys():
        print("\t"+key)

    avgs = get_averages(data)
    for key in avgs.keys():
        print(key)
        print("\tacc :",avgs[key][0])
        print("\tlogl:",avgs[key][1])
        print("\tf1  :",avgs[key][2])


if __name__ == "__main__":
    main()

