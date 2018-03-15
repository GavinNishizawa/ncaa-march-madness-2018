from sklearn import neighbors


def create():
    return neighbors.KNeighborsClassifier(n_neighbors=5)

