import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_data import load_data


def seed_val(seed):
    seed_map = ["W","X","Y","Z"]
    first_letter = seed[:1]
    seednumber = int(seed[1:3])

    # the magic formula to convert seed to a numerical value
    return seed_map.index(first_letter) + 1 + (seednumber-1)*4


def get_seed_data(data):
    seedData = data["NCAATourneySeeds"]
    seasonResults = data["AllCompactResults"]

    # join NCAATourneySeeds on AllCompactResults where TeamID=WTeamID and Season=Season
    r_data = seedData.merge(seasonResults, left_on=["TeamID","Season"], right_on=["WTeamID","Season"])

    # join r_data on NCAATourneySeeds where LTeamID=TeamID and Season=Season
    r_data = r_data.merge(seedData, left_on=["LTeamID","Season"], right_on=["TeamID","Season"])

    # convert seeds to seed values
    r_data["Seed_xv"] = r_data["Seed_x"].apply(seed_val)
    r_data["Seed_yv"] = r_data["Seed_y"].apply(seed_val)
    r_data["WLoc"] = r_data["WLoc"].apply(
        lambda x: 1 if x == 'H' else 0)
    r_data["Avg_score_x"] = r_data["TeamID_x"].apply(
            lambda x:
            r_data[r_data["TeamID_x"] == x]["WScore"].mean())
    r_data["Avg_score_y"] = r_data["TeamID_y"].apply(
            lambda x:
            r_data[r_data["TeamID_y"] == x]["LScore"].mean())

    # trim to wanted values
    r_data = r_data[["Seed_xv",
        "Seed_yv",
        "WLoc",
        "TeamID_x",
        "TeamID_y",
        "Avg_score_x",
        "Avg_score_y",
        ]]

    # copy to r2 data for inverse results
    r2_data = pd.DataFrame()
    r2_data["Seed_xv"] = r_data["Seed_yv"]
    r2_data["Seed_yv"] = r_data["Seed_xv"]
    r2_data["TeamID_x"] = r_data["TeamID_y"]
    r2_data["TeamID_y"] = r_data["TeamID_x"]
    r2_data["Avg_score_x"] = r_data["Avg_score_y"]
    r2_data["Avg_score_y"] = r_data["Avg_score_x"]
    r2_data["WLoc"] = r_data["WLoc"].apply(
        lambda x: 0 if x == 1 else 1)

    # append column of 0s for target
    r2_data = r2_data.assign(results=0)

    # append column of 1s for target
    r_data = r_data.assign(results=1)

    # concat win and loss data
    return pd.concat([r_data, r2_data])


def main():
    data = load_data()
    seed_data = get_seed_data(data)
    print(seed_data)


if __name__=="__main__":
    main()
