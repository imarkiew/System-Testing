import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot as plt
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from PyFiles import FuzzyAlgorithms

def find_categories(y):
    return list(set(y))

def match_categories(category, y):
    return [1 if _y == category else 0 for _y in y]

def find_min_max(X):
    return np.min(X, axis=0), np.max(X, axis=0)

def find_expanded_min_max(X, delta):
    min, max = find_min_max(X)
    return [i - delta for i in min], [i + delta for i in max]

def find_subset_by_category(X, y, category):
    return [x for x, _y in zip(X, y) if _y == category]

def RMSE(y_1, y_2):
    return sqrt(mean_squared_error(y_1, y_2))

def transform_parameters_to_indyvidual(list_of_parameters):
    indyvidual = []
    for col in list_of_parameters:
        indyvidual.extend(col)
    return indyvidual

def transform_indyvidual_to_parameters(indyvidual):
    number_of_parameters = 2
    list_of_parameters = []
    for i in range(len(indyvidual)//number_of_parameters):
        b = indyvidual[number_of_parameters*i]
        c = indyvidual[number_of_parameters*i + 1]
        list_of_parameters.append([b, c])
    return list_of_parameters

def accuracy(y_1, y_2):
    counter = 0
    for yy_1, yy_2 in zip(y_1, y_2):
        if yy_1 == yy_2:
            counter += 1
    return counter/len(y_1)

def find_avg_of_vectors_by_column(vectors):
    return np.mean(vectors, axis=0)

def plot_errors(errors, is_plot_saved, name_of_plot):
    epochs = range(1, len(errors[0]) + 1)
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(epochs, errors[0], "b-", label="Średni błąd na zbiorach trenujących")
    ax.plot(epochs, errors[1], "r-", label="Średni błąd na zbiorach testowych")
    plt.xlabel('iteracja', fontsize=15)
    plt.ylabel('Średnie RMSE', fontsize=15)
    ax.legend()
    if is_plot_saved:
        fig.savefig(name_of_plot, bbox_inches='tight')
    plt.show()

def plot_accuracies(accuracies, is_plot_saved, name_of_plot):
    epochs = range(1, len(accuracies[0]) + 1)
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(epochs, accuracies[0], "b-", label="Średnia celność predykcji na zbiorach trenujących")
    ax.plot(epochs, accuracies[1], "r-", label="Średnia celność predykcji na zbiorach testowych")
    plt.xlabel('iteracja', fontsize=15)
    plt.ylabel('Średnie wartość predykcji', fontsize=15)
    ax.legend()
    if is_plot_saved:
        fig.savefig(name_of_plot, bbox_inches='tight')
    plt.show()

def plot_scores(scores, is_plot_saved, name_of_plot):
    epochs = range(1, len(scores[0]) + 1)
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(epochs, scores[0], "b-", label="Średnia wartość współczynnika Kappa Cohena na zbiorach trenujących")
    ax.plot(epochs, scores[1], "r-", label="Średnia wartość współczynnika Kappa Cohena na zbiorach testowych")
    plt.xlabel('iteracja', fontsize=15)
    plt.ylabel('Średnie wartość współczynnika Kappa Cohena', fontsize=15)
    ax.legend()
    if is_plot_saved:
        fig.savefig(name_of_plot, bbox_inches='tight')
    plt.show()

def MMC(y_1, y_2):
    return matthews_corrcoef(y_1, y_2)

def CKS(y_1, y_2):
    return cohen_kappa_score(y_1, y_2)

def run_test(Xx, Xt, yy, yt, absolute_path):
    path = absolute_path + "/PyResults/"
    parameters_and_categories, train_errors, test_errors = FuzzyAlgorithms.learn_system(Xx, yy, Xt, yt)
    predictions = [
        FuzzyAlgorithms.run_system(Xx, [[rule[0][i], rule[1], rule[2], rule[3]] for rule in parameters_and_categories])
        for i in range(len(parameters_and_categories[0][0]))]
    train_accuracies_for_epochs = [accuracy(predicion, yy) for predicion in predictions]
    train_scores_for_epochs = [CKS(predicion, yy) for predicion in predictions]
    predictions = [
        FuzzyAlgorithms.run_system(Xt, [[rule[0][i], rule[1], rule[2], rule[3]] for rule in parameters_and_categories])
        for i in range(len(parameters_and_categories[0][0]))]
    test_accuracies_for_epochs = [accuracy(predicion, yt) for predicion in predictions]
    test_scores_for_epochs = [CKS(predicion, yt) for predicion in predictions]
    pd.DataFrame(train_accuracies_for_epochs).to_csv(path + "train_acc", index=False, header=False, mode="a")
    pd.DataFrame(test_accuracies_for_epochs).to_csv(path + "test_acc", index=False, header=False, mode="a")
    pd.DataFrame(train_scores_for_epochs).to_csv(path + "train_scores", index=False, header=False, mode="a")
    pd.DataFrame(test_scores_for_epochs).to_csv(path + "test_scores", index=False, header=False, mode="a")
    pd.DataFrame(find_avg_of_vectors_by_column(train_errors)).to_csv(path + "train_RMSE", index=False, header=False, mode="a")
    pd.DataFrame(find_avg_of_vectors_by_column(test_errors)).to_csv(path + "test_RMSE", index=False, header=False, mode="a")
    pd.DataFrame([test_accuracies_for_epochs[-1]]).to_csv(path + "acc_py", index=False, header=False, mode="a")
    pd.DataFrame([test_scores_for_epochs[-1]]).to_csv(path + "scores_py", index=False, header=False, mode="a")