import numpy as np
from PyFiles import Tools, EvolutionStrategy

def pi_function(x, a, b, c, d):
    value = 0
    if a <= x and 0.5*(a + b) > x:
        value = 2*((x - a)/(b - a))**2
    elif 0.5*(a + b) <= x and b > x:
        value = 1 - 2*((x - b)/(b - a))**2
    elif b <= x and x < c:
        value = 1
    elif c <= x and 0.5*(c + d) > x:
        value = 1 - 2*((x - c)/(d - c))**2
    elif 0.5*(c + d) <= x and d > x:
        value = 2*((x - d)/(d - c))**2
    return value

def aggregation_operator(values):
    return np.mean(values)

def aggregated_output(X, parameters, min, max):
    output = []
    for xx in X:
        values = []
        for col, min_col, max_col, param in zip(xx, min, max, parameters):
            values.append(pi_function(col, min_col, param[0], param[1], max_col))
        output.append(aggregation_operator(values))
    return output

def learn_system(X, y, Xt, yt):
    delta = 0.01
    cxpb = 0.5
    mutpb = 0.1
    start_population_size = 25
    size_of_offspring = 50
    number_of_epochs = 50
    categories = Tools.find_categories(y)
    parameters_and_categories = []
    hof_errors_for_categories = []
    test_errors_for_categories = []
    for category in categories:
        print("Learning for category {}".format(category))
        train_subset = Tools.find_subset_by_category(X, y, category)
        train_min, train_max = Tools.find_expanded_min_max(train_subset, delta)
        train_y_bin = Tools.match_categories(category, y)
        test_y_bin = Tools.match_categories(category, yt)
        hofs, train_errors, test_errors = EvolutionStrategy.run_evolution_strategy(X, train_y_bin, Xt, test_y_bin, train_min, train_max, cxpb, mutpb, start_population_size,
                                                                                  size_of_offspring, number_of_epochs)
        print("\n")
        parameters_and_categories.append([[Tools.transform_indyvidual_to_parameters(hof) for hof in hofs], train_min, train_max, category])
        hof_errors_for_categories.append(train_errors)
        test_errors_for_categories.append(test_errors)
    return parameters_and_categories, hof_errors_for_categories, test_errors_for_categories

def run_system(X, parameters_and_categories):
    prediction = []
    for xx in X:
        output = []
        for rule in parameters_and_categories:
            parameters = rule[0]
            min = rule[1]
            max = rule[2]
            output.append(aggregated_output([xx], parameters, min, max)[0])
        index = output.index(np.max(output))
        prediction.append(parameters_and_categories[index][3])
    return prediction








