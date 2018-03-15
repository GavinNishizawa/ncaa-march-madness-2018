import numpy as np
from read_data import load_data
from save_data import save_object, load_object
from sklearn import metrics
import matplotlib.pyplot as plt
from seed_pred import get_seed_data
from split_data import split_data
from get_models import get_models
models = get_models()
from models import voting_hard, voting_soft


def plot(X, Y, c_fn):
    # define colors array
    colors = np.array([c_fn(v) for v in Y])

    plt.scatter(X[:,0], X[:,1], c=colors, s=0.5)
    plt.tight_layout()


def plot_model(train, test, model_test_output):
    # plot training data in blue and orange
    c_fn = lambda v: "blue" if v == 1 else "orange"
    plot(train[:, :2], train[:, 2], c_fn)
    plt.show()
    plt.close()

    # plot test data in green and red
    c_fn = lambda v: "green" if v == 1 else "red"
    plot(test[:, :2], model_test_output, c_fn)

    plt.show()
    plt.close()


def run_model(m_name, train_data, test_data):
    if m_name == "voting_hard":
        model = voting_hard
    elif m_name == "voting_soft":
        model = voting_soft
    else:
        model = models[m_name]

    # train model on training data
    trained_model = model.train(train_data)
    test_results = model.test(test_data[:,:2], trained_model)

    # calc accuracy
    test_accuracy = metrics.accuracy_score(test_data[:,2], test_results)
    test_precision = metrics.precision_score(test_data[:,2], test_results)
    test_recall = metrics.recall_score(test_data[:,2], test_results)
    test_f1 = metrics.f1_score(test_data[:,2], test_results)
    test_log_loss = metrics.log_loss(test_data[:,2], test_results)
    test_report = metrics.classification_report(test_data[:,2], test_results)

    print("\nResults for",m_name,":")
    print("\tLog loss : %f" % test_log_loss)
    print("\tAccuracy : %0.5f" % test_accuracy)
    #print("\tRecall   : %0.5f" % test_recall)
    #print("\tPrecision: %0.5f" % test_precision)
    print("\tF1 score : %0.5f" % test_f1)
    print("\nReport:\n", test_report,"\n")

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

    #train, test, test_predict = run_model("knn", train_data, test_data)
    #plot_model(train, test, test_predict)
    #run_model("voting_hard", train_data, test_data)
    #run_model("voting_soft", train_data, test_data)


if __name__ == "__main__":
    main()

