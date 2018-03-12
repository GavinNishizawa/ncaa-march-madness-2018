import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_data import load_data

data = load_data()
def main():
    data = load_data()
    id_to_region(data,1385,1985)

def id_to_region(data, id, year):
    x = ["W","X","Y","Z"]
    seedData = data["NCAATourneySeeds"]

    currRow = seedData[(seedData['Season'] == year) & (seedData['TeamID'] ==id)][:1]
    currSeed = currRow['Seed'].values[0]

    first_letter = currSeed[:1]
    seednumber = int(currSeed[1:3])

    # the magic formula to convert seed to a numerical value
    return x.index(first_letter) + 1 + (seednumber-1)*4

if __name__=="__main__":
    main()
