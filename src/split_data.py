import numpy as np
import random as r


def get_shuffled_inds(n):
    # create a list of indicies
    inds = list(range(n))

    # shuffle inds
    #r.seed(9001)
    for i in range(r.randint(3,10)):
        r.shuffle(inds)

    return inds


def split_data(data, ratio):
    '''
    Note: each win data point must be in the same training/testing set as the corresponding loss data point, otherwise the testing and training sets have shared data.
    '''

    # split data into wins and losses assume wins then losses
    ld = int(len(data)/2)
    wins = data[:ld]
    losses = data[ld:]

    # create a list of indicies
    inds = get_shuffled_inds(ld)

    # map the wins and losses to the shuffled locations
    # this should maintain the index between w and l points
    shf_wins = [wins[inds.index(i)] for i in range(ld)]
    shf_losses = [losses[inds.index(i)] for i in range(ld)]

    # split each set into sections according to ratio
    s_ind_w = int(len(shf_wins)*ratio)
    s_ind_l = int(len(shf_losses)*ratio)

    # create a training and testing set with wins and losses
    train = np.vstack((shf_wins[:s_ind_w], shf_losses[:s_ind_l]))
    test = np.vstack((shf_wins[s_ind_w:], shf_losses[s_ind_l:]))

    # get a shuffled index map for train and test
    # note: r.shuffle(train) did not work as intended
    train_is = get_shuffled_inds(s_ind_w+s_ind_l)
    test_is = get_shuffled_inds(ld-s_ind_w-s_ind_l)

    # map the train and test data to the shuffled locations
    shf_train = [train[train_is.index(i)] for i in range(len(train_is))]
    shf_test = [test[test_is.index(i)] for i in range(len(test_is))]

    return train, test


