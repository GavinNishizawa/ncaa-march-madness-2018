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


'''
TODO: fix bugs and clean up
'''
def get_test_data(data):
    seedData = data["NCAATourneySeeds"]
    seedData2018 = seedData[seedData["Season"] == 2018]
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
    sD2018 = sD2018.merge(seedData2018, left_on=["TeamID_x"], right_on=["TeamID"])
    sD2018 = sD2018[["TeamID_x","TeamID_y","Season","Seed"]]
    sD2018 = sD2018.merge(seedData2018,\
            left_on=["TeamID_y","Season"],\
            right_on=["TeamID","Season"])
    sD2018 = sD2018[["TeamID_x","TeamID_y",
        "Season","Seed_x","Seed_y"]]

    # convert seeds to seed values
    sD2018["Seed_xv"] = sD2018["Seed_x"].apply(seed_val)
    sD2018["Seed_yv"] = sD2018["Seed_y"].apply(seed_val)
    print("sD2018 shape:",sD2018.shape)


    seasonResults = data["AllCompactResults"]

    r_data = seasonResults
    r_data["WLoc"] = r_data["WLoc"].apply(
        lambda x: 1 if x == 'H' else 0)
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
    print("r_data:",r_data)


    sD2018 = sD2018.merge(r_data, left_on=["TeamID_x"], right_on=["WTeamID"])
    print(sD2018)

    # trim columns
    fr_data = sD2018[[
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
        "Avg_score_y"
        ]]

    return fr_data


'''
TODO: fix bugs and clean up
'''
def main():
    data = load_data()
    seed_data = get_seed_data(data)
    train_data = seed_data.values

    td_fn = os.path.join("data","test_predict_data")
    # load test_data if it exists
    test_data = load_object(td_fn)

    if type(test_data) == type(None):
        test_data = get_test_data(data)
        save_object(td_fn, test_data)

    print("tdata test:",test_data.shape)

    t_data = {
        "train_target": train_data[:, -1],
        "train_data": train_data[:, :-1],
        "test_data": test_data.values
        }

    t_data = apply_pca(t_data)
    ps = run_prediction("knn", t_data)
    print("ps",ps.shape)
    ps = np.array(ps)
    ps_df = pd.DataFrame(ps[:,0])

    data_2018 = test_data
    data_2018["prediction"] = ps_df[0]
    data_2018["id"] = data_2018.apply(
            lambda r: "2018_"+
                str(int(r["TeamID_x"]))+"_"+
                str(int(r["TeamID_y"])), axis=1)

    data_2018 = data_2018[["id","prediction"]]
    data_2018 = data_2018.drop_duplicates()
    print(data_2018)

    data_2018.to_csv("predictions_2018.csv", index=False)
    print(data_2018)


if __name__=="__main__":
    main()
