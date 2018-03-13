import random as r
import numpy as np
from read_data import load_data
from save_data import save_object, load_object
from sklearn import neighbors, metrics
import matplotlib.pyplot as plt
from seed_pred import get_seed_data


def plot(X, Y, c_fn):
    # define colors array
    colors = np.array([c_fn(v) for v in Y])

    plt.scatter(X[:,0], X[:,1], c=colors, s=0.5)
    plt.tight_layout()


def train(data):
    X = data[:, :2]
    Y = data[:, 2]
    model = neighbors.KNeighborsClassifier(n_neighbors=5)
    model.fit(X, Y)
    return model


def test(test_data, model):
    return model.predict(test_data)


def split_data(data, ratio):

    # split data into wins and losses assume wins then losses
    ld = int(len(data)/2)
    wins = data[:ld]
    losses = data[ld:]

    # shuffle data
    r.seed(9001)
    for i in range(r.randint(3,10)):
        r.shuffle(wins)
        r.shuffle(losses)

    # split each set into sections according to ratio
    s_ind_w = int(len(wins)*ratio)
    s_ind_l = int(len(losses)*ratio)

    # create a training and testing set with wins and losses
    train = np.vstack((wins[:s_ind_w], losses[:s_ind_l]))
    test = np.vstack((wins[s_ind_w:], losses[s_ind_l:]))

    # shuffle the wins and losses in each set
    r.shuffle(train)
    r.shuffle(test)

    return train, test


def main():
    data = load_data()
    seed_data = get_seed_data(data)
    # split data into training 70% and testing 30%
    train_data, test_data = split_data(seed_data.values, 0.7)

    # plot training data in blue and orange
    c_fn = lambda v: "blue" if v == 1 else "orange"
    plot(train_data[:, :2], train_data[:, 2], c_fn)

    # train model on training data
    model = train(train_data)
    t_results = test(test_data[:,:2], model)

    # calc accuracy
    accuracy = metrics.accuracy_score(test_data[:,2], t_results)
    print("Accuracy:",accuracy)

    # plot test data in green and red
    c_fn = lambda v: "green" if v == 1 else "red"
    plot(test_data[:, :2], t_results, c_fn)

    plt.show()


if __name__ == "__main__":
    main()
