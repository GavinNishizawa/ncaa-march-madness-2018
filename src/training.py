import numpy as np
from read_data import load_data
from save_data import save_object, load_object
from sklearn import neighbors
import matplotlib.pyplot as plt
from seed_pred import get_seed_data

n_neighbors = 5
scope = load_data()
fake_data = np.array([
    [12, 23, 34, 45, 56],
    [21, 32, 43, 54, 65]
]).T


def plot(data):
    print(data)
    data = data.values
    X = data[:, :2]
    colors = np.array(["blue" if v == 1 else "red" for v in X[:,2]])
    # print(X[0])
    # print(X[1])
    Y = data[:, 2]
    # model = neighbors.KNeighborsClassifier(n_neighbors=5)
    # model.fit(X, Y)



    plt.scatter(X[:,0], X[:,1], c=colors, s=0.5)
    # clf = neighbors.KNeighborsClassifier(n_neighbors)
    # clf.fit(X, y)
    plt.tight_layout()
    plt.show()


def main():
    data = load_data()
    seed_data = get_seed_data(data)


    plot(seed_data)



if __name__ == "__main__":
    main()
