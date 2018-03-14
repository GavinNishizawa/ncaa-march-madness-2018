import knn
import svm
import log_reg
import perceptron
import pass_aggr
import huber
import gaussian_nb
import multinomial_nb
import decision_tree
import random_forest
import extra_trees
import bag_svm
import bag_gaussian_nb
import ada_boost
import gradient_boosting


models = {
    "knn": knn, "svm": svm,
    "log_reg": log_reg,
    "bag_svm": bag_svm,
    "huber": huber,
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

