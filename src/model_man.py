from models import knn, svm, log_reg, perceptron, \
pass_aggr, huber, gaussian_nb, multinomial_nb, \
decision_tree, random_forest, extra_trees, bag_svm, \
bag_gaussian_nb, ada_boost, gradient_boosting


models = {
    "knn": knn, "svm": svm,
    "log_reg": log_reg,
    "bag_svm": bag_svm,
    #"huber": huber,
    "gaussian_nb": gaussian_nb,
    "bag_gaussian_nb": bag_gaussian_nb,
    "multinomial_nb": multinomial_nb,
    "perceptron": perceptron,
    "decision_tree": decision_tree,
    "random_forest": random_forest,
    "extra_trees": extra_trees,
    "ada_boost": ada_boost,
    "gradient_boosting": gradient_boosting,
    "pass_aggr": pass_aggr
}



def get_models():
    return models.copy()


def train(model, train_data, train_target):
    model.fit(train_data, train_target)
    return model


def test(model, test_data):
    return model.predict(test_data)

