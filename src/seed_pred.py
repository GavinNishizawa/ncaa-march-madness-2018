import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_data import load_data


def main():
    data = load_data()
    seed_data = get_seed_data(data)
    print(seed_data)


def seed_val(seed):
    seed_map = ["W","X","Y","Z"]
    first_letter = seed[:1]
    seednumber = int(seed[1:3])

    # the magic formula to convert seed to a numerical value
    return seed_map.index(first_letter) + 1 + (seednumber-1)*4


def get_seed_data(data):
    seedData = data["NCAATourneySeeds"]
    seasonResults = data["AllCompactResults"]

    # join NCAATourneySeeds on RegularSeasonCompactResults where TeamID=WTeamID and Season=Season
    r_data = seedData.merge(seasonResults, left_on=["TeamID","Season"], right_on=["WTeamID","Season"])

    # join r_data on NCAATourneySeeds where LTeamID=TeamID and Season=Season
    r_data = r_data.merge(seedData, left_on=["LTeamID","Season"], right_on=["TeamID","Season"])

    # convert seeds to seed values
    r_data["Seed_xv"] = r_data["Seed_x"].apply(seed_val)
    r_data["Seed_yv"] = r_data["Seed_y"].apply(seed_val)

    # trim to only seed values
    r_data = r_data[["Seed_xv","Seed_yv"]]

    # copy to r2 data for inverse results
    r2_data = pd.DataFrame()
    r2_data["Seed_xv"] = r_data["Seed_yv"]
    r2_data["Seed_yv"] = r_data["Seed_xv"]

    # append column of 0s for target
    r2_data = r2_data.assign(results=0)

    # append column of 1s for target
    r_data = r_data.assign(results=1)

    # concat win and loss data
    return pd.concat([r_data, r2_data])


if __name__=="__main__":
    main()
