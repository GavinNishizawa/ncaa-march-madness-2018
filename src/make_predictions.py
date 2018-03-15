import os
import itertools as itr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_data import load_data
from save_data import save_object, load_object
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
    print(seedData2018)
    sD2018 = pd.DataFrame(list(itr.combinations( seedData2018.TeamID, 2)))
    sD2018.columns = ["TeamID_l","TeamID_r"]
    sD2018["TeamID_x"] = sD2018.apply(
            lambda r: int(r["TeamID_l"]) \
                if int(r["TeamID_l"]) < int(r["TeamID_r"]) \
                else int(r["TeamID_r"]), axis=1)
    sD2018["TeamID_y"] = sD2018.apply(
            lambda r: int(r["TeamID_r"]) \
                if int(r["TeamID_l"]) < int(r["TeamID_r"]) \
                else int(r["TeamID_l"]), axis=1)
    sD2018 = sD2018[["TeamID_x","TeamID_y"]]
    sD2018 = sD2018.sort_values(["TeamID_x","TeamID_y"])
    print(sD2018)
    sD2018 = sD2018.merge(seedData2018, left_on=["TeamID_x"], right_on=["TeamID"])
    sD2018 = sD2018[["TeamID_x","TeamID_y","Season","Seed"]]
    print("sD:",sD2018)
    print("seed:",seedData2018)
    sD2018 = sD2018.merge(seedData2018,\
            left_on=["TeamID_y","Season"],\
            right_on=["TeamID","Season"])
    print(sD2018)
    sD2018 = sD2018[["TeamID_x","TeamID_y",
        "Season","Seed_x","Seed_y"]]

    # convert seeds to seed values
    sD2018["Seed_xv"] = sD2018["Seed_x"].apply(seed_val)
    sD2018["Seed_yv"] = sD2018["Seed_y"].apply(seed_val)


    seasonResults = data["AllCompactResults"]

    '''
    # join NCAATourneySeeds on CompactResults where TeamID=WTeamID and Season=Season
    r_data = seedData.merge(seasonResults, left_on=["TeamID","Season"], right_on=["WTeamID","Season"])

    # join r_data on NCAATourneySeeds where LTeamID=TeamID and Season=Season
    r_data = r_data.merge(seedData, left_on=["LTeamID","Season"], right_on=["TeamID","Season"])
    '''

    r_data = seasonResults
    r_data["WLoc"] = r_data["WLoc"].apply(
        lambda x: 1 if x == 'H' else 0)
    print("="*20,"r_data")
    print("="*20,"r_data")
    print(r_data)
    r_data["Avg_WLoc_x"] = r_data["WTeamID"].apply(
            lambda x:
            r_data[r_data["WTeamID"] == x]["WLoc"].mean())
    r_data["Avg_WLoc_y"] = r_data["LTeamID"].apply(
            lambda x:
            r_data[r_data["WTeamID"] == x]["WLoc"].mean())
    r_data["Avg_score_x"] = r_data["WTeamID"].apply(
            lambda x:
            r_data[r_data["WTeamID"] == x]["WScore"].mean())
    r_data["Avg_score_y"] = r_data["LTeamID"].apply(
            lambda x:
            r_data[r_data["WTeamID"] == x]["WScore"].mean())
    print(r_data)


    sD2018 = sD2018.merge(r_data, left_on=["TeamID_x","TeamID_y"], right_on=["WTeamID","LTeamID"])
    print(sD2018)

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

    td_fn = os.path.join("data","test_predict_data")
    # load test_data if it exists
    test_data = load_object(td_fn)

    if test_data == None:
        test_data = get_test_data(data)
        save_object(td_fn, test_data)

    t_data = {
        "train_target": train_data[:, -1],
        "train_data": train_data[:, :-1],
        "test_data": test_data.values
        }

    t_data = apply_pca(t_data)
    ps = run_prediction("knn", t_data)
    ps = np.array(ps)
    print(ps.shape)
    data_2018["prediction"] = pd.DataFrame(ps[:,0])
    data_2018["id"] = data_2018.apply(
            lambda r: "2018_"+
                str(int(r["TeamID_x"]))+"_"+
                str(int(r["TeamID_y"])), axis=1)

    data_2018 = data_2018[["id","prediction"]]

    print(data_2018)
    data_2018.to_csv("predictions_2018.csv", index=False)


if __name__=="__main__":
    main()
