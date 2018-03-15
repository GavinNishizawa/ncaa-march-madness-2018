import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_data import load_data
from run_model import apply_pca, run_prediction
from seed_pred import get_seed_data


def seed_val(seed):
    seed_map = ["W","X","Y","Z"]
    first_letter = seed[:1]
    seednumber = int(seed[1:3])

    # the magic formula to convert seed to a numerical value
    return seed_map.index(first_letter) + 1 + (seednumber-1)*4


def get_test_data(data):
    seedData = data["NCAATourneySeeds"]
    seedData2018 = seedData[seedData["Season"] == 2018]
    seasonResults = data["AllCompactResults"]

    # join NCAATourneySeeds on CompactResults where TeamID=WTeamID and Season=Season
    r_data = seedData.merge(seasonResults, left_on=["TeamID","Season"], right_on=["WTeamID","Season"])

    # join r_data on NCAATourneySeeds where LTeamID=TeamID and Season=Season
    r_data = r_data.merge(seedData, left_on=["LTeamID","Season"], right_on=["TeamID","Season"])

    # convert seeds to seed values
    r_data["Seed_xv"] = r_data["Seed_x"].apply(seed_val)
    r_data["Seed_yv"] = r_data["Seed_y"].apply(seed_val)
    r_data["WLoc"] = r_data["WLoc"].apply(
        lambda x: 1 if x == 'H' else 0)
    r_data["Avg_WLoc_x"] = r_data["TeamID_x"].apply(
            lambda x:
            r_data[r_data["TeamID_x"] == x]["WLoc"].mean())
    r_data["Avg_WLoc_y"] = r_data["TeamID_y"].apply(
            lambda x:
            r_data[r_data["TeamID_y"] == x]["WLoc"].mean())
    r_data["Avg_score_x"] = r_data["TeamID_x"].apply(
            lambda x:
            r_data[r_data["TeamID_x"] == x]["WScore"].mean())
    r_data["Avg_score_y"] = r_data["TeamID_y"].apply(
            lambda x:
            r_data[r_data["TeamID_y"] == x]["LScore"].mean())


    r_data = r_data.merge(seedData2018, left_on=["Seed_x","TeamID_x"], right_on=["Seed","TeamID"])


    # trim to wanted values
    r_data = r_data[[
        #"Season",
        #"DayNum",
        #"NumOT",
        "Seed_xv",
        "Seed_yv",
        #"WLoc",
        "Avg_WLoc_x",
        "Avg_WLoc_y",
        "TeamID_x",
        "TeamID_y",
        "Avg_score_x",
        "Avg_score_y",
        ]]

    return r_data


def main():
    data = load_data()
    seed_data = get_seed_data(data)
    train_data = seed_data.values
    print(train_data.shape)

    test_data = get_test_data(data).values
    print(test_data.shape)

    t_data = {
        "train_target": train_data[:, -1],
        "train_data": train_data[:, :-1],
        "test_data": test_data
        }

    t_data = apply_pca(t_data)
    ps = run_prediction("knn", t_data)
    print(ps)


if __name__=="__main__":
    main()
