import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_data import load_data

data = load_data()
def main():
    data = load_data()
    #print(data["Seasons"].plot())
    #plt.show()
    #print(data["Seasons"]["RegionZ"])
    #data["RegularSeasonCompactResults"]["WTeamID"]
    #data["RegularSeasonCompactResults"]["LTeamID"]
    id_to_region(data,1385,1985)

def id_to_region(data, id, year):
    seedData = data["NCAATourneySeeds"]
    currRow = seedData[(seedData['Season'] == year) & (seedData['TeamID'] ==id)][:1]
    print(currRow)
    x = ["W","X","Y","Z"]
    #currSeed = currRow['Seed'][0]
    #print(currSeed)
    #first_letter,seednumber = currSeed[:1], int(currSeed[1:3])
    #print(x.index(first_letter)*16 + seednumber)

if __name__=="__main__":
    main()
