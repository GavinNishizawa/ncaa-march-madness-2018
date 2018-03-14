import random as r
import numpy as np
from read_data import load_data
from save_data import save_object, load_object
from sklearn import metrics
import matplotlib.pyplot as plt
from seed_pred import get_seed_data
from models import models
import voting


def plot(X, Y, c_fn):
    # define colors array
    colors = np.array([c_fn(v) for v in Y])

    plt.scatter(X[:,0], X[:,1], c=colors, s=0.5)
    plt.tight_layout()


def split_data(data, ratio):
    '''
    Note: each win data point must be in the same training/testing set as the corresponding loss data point, otherwise the testing and training sets have shared data.
    '''

    # split data into wins and losses assume wins then losses
    ld = int(len(data)/2)
    wins = data[:ld]
    losses = data[ld:]

    # create a list of indicies
    inds = list(range(ld))

    # shuffle inds
    #r.seed(9001)
    for i in range(r.randint(3,10)):
        r.shuffle(inds)

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

    # shuffle the wins and losses in each set
    r.shuffle(train)
    r.shuffle(test)

    return train, test


def plot_model(train, test, model_test_output):
    # plot training data in blue and orange
    c_fn = lambda v: "blue" if v == 1 else "orange"
    plot(train[:, :2], train[:, 2], c_fn)

    # plot test data in green and red
    c_fn = lambda v: "green" if v == 1 else "red"
    plot(test[:, :2], model_test_output, c_fn)

    plt.show()


def run_model(m_name, train_data, test_data):
    if m_name != "voting":
        model = models[m_name]
    else:
        model = voting

    # train model on training data
    trained_model = model.train(train_data)
    test_results = model.test(test_data[:,:2], trained_model)

    # calc accuracy
    test_accuracy = metrics.accuracy_score(test_data[:,2], test_results)
    print("Accuracy for",m_name,":", test_accuracy)

    return train_data, test_data, test_results


def main():
    # create training/testing dataset
    data = load_data()
    seed_data = get_seed_data(data)
    # split data into training 70% and testing 30%
    train_data, test_data = split_data(seed_data.values, 0.7)

    # run training and testing of each of the models on the dataset
    for key in models.keys():
        run_model(key, train_data, test_data)
    #plot_model(train, test, test_predict)

    # TODO: fix voting prediction with huber
    run_model("voting", train_data, test_data)


if __name__ == "__main__":
    main()

