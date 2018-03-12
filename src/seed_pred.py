import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_data import load_data

data = load_data()
def main():
    data = load_data()
    finalList = []
    seasonResults = data["RegularSeasonCompactResults"]
    #print(seasonResults[:1])
    df = pd.DataFrame([get_seed_list(seasonResults[:1],data)])
    df.columns = ["w_team","l_team"]
    print( df["w_team"] )


#Returns list [WinningteamSeedNum, LosingteamSeedNum]
def get_seed_list(seasonResultsRow, data):
    list = []
    winner = seasonResultsRow["WTeamID"].values[0]
    loser = seasonResultsRow["LTeamID"].values[0]
    year = seasonResultsRow['Season'].values[0]
    list.append(id_to_region(data,winner,year))
    list.append(id_to_region(data,loser,year))
    return list

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
