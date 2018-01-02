import numpy as np
from deap import base, creator, tools, algorithms
import random
from PyFiles import FuzzyAlgorithm, Tools

def initIndividual(icls, content):
    return icls(content)

def initPopulation(pcls, ind_init, min, max, size_of_initial_population):
    contents = []
    for i in range(size_of_initial_population):
        param = []
        for j in range(len(min)):
            b = random.uniform(min[j], max[j])
            c = random.uniform(b, max[j])
            param.append([b, c])
        contents.append(Tools.transform_parameters_to_indyvidual(param))
    return pcls(ind_init(c) for c in contents)

def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            delta = 0.01
            for child in offspring:
               for i in range(len(min)):
                   if child[2*i + 0] > max[i]:
                       child[2*i + 0] = max[i] - delta*abs(max[i])
                   elif child[2*i + 0] < min[i]:
                       child[2*i + 0] = min[i] + delta*abs(min[i])
                   if child[2*i + 1] < child[2*i + 0]:
                       child[2*i + 1] = child[2*i + 0] + delta*abs(child[2*i + 0])
                   elif child[2*i + 1] > max[i]:
                       child[2*i + 1] = max[i] - delta*abs(max[i])
            return offspring
        return wrapper
    return decorator

def update_loss_of_indyvidual(indyvidual, X, y_bin, min, max, offspring, hof, is_update_enabled):
    parameters = Tools.transform_indyvidual_to_parameters(indyvidual)
    output = FuzzyAlgorithm.aggregated_output(X, parameters, min, max)
    rmse = Tools.RMSE(output, y_bin)
    if is_update_enabled:
        indyvidual.fitness.values = rmse,
        hof.update(offspring)
    return rmse

def run_evolution_strategy(X, train_y_bin, Xt, test_y_bin, train_min, train_max, cxpb, mutpb, start_population_size,
                                                size_of_offspring, number_of_epochs):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("individual_guess", initIndividual, creator.Individual)
    toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess)
    toolbox.register("mate", tools.cxSimulatedBinary, eta=1)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.decorate("mate", checkBounds(train_min, train_max))
    toolbox.decorate("mutate", checkBounds(train_min, train_max))
    population = toolbox.population_guess(train_min, train_max, start_population_size)
    hof = tools.HallOfFame(1)
    avg_error_on_population = []
    hof_errors = []
    test_errors = []
    hofs = []
    for i in range(number_of_epochs):
        indyvidual_errors = []
        offspring = algorithms.varOr(population, toolbox, size_of_offspring, cxpb, mutpb)
        for indyvidual in offspring:
            indyvidual_errors.append(update_loss_of_indyvidual(indyvidual, X, train_y_bin, train_min, train_max, offspring, hof, True))
        population[:] = tools.selBest(offspring, start_population_size)
        avg_error_on_population.append(np.mean(indyvidual_errors))
        hof_rmse = update_loss_of_indyvidual(hof[0], X, train_y_bin, train_min, train_max, offspring, hof, False)
        hof_errors.append(hof_rmse)
        test_error = update_loss_of_indyvidual(hof[0], Xt, test_y_bin, train_min, train_max, offspring, hof, False)
        test_errors.append(test_error)
        hofs.append(hof[0])
        print("Epoch : {} avg RMSE for population : {} hof : {}".format(i + 1, avg_error_on_population[-1], hof[0]))
    return hofs, hof_errors, test_errors
