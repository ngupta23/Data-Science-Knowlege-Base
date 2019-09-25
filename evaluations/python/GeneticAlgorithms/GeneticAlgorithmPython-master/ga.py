import numpy as np

"""
Fixes Needed:
(1) The same chromosomes are mutated every time. This can lead to issues
    Maybe this should be made random (DONE)
(2) The mutated values may over time drift and go beyond the intial range
    of the parameter. (DONE by resampling from correct space)
(3) The search space is hard coded here. Need to make it generic (TBD)
"""


def cal_pop_fitness(equation_inputs, pop):
    """
    This is the inverse of the loss function which you would typically minimize
    """
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input
    # and its corresponding weight.
    fitness = np.sum(pop*equation_inputs, axis=1)
    return fitness


def select_mating_pool_org(pop, fitness, num_parents):
    """
    Original function
    """
    # Selecting the best individuals in the current generation as parents
    # for producing the offspring of the next generation.
    parents = np.empty((num_parents, pop.shape[1]))
    # Need to create a copy otherwise the global variable gets affected.
    loFiteness = fitness.copy()

    for parent_num in range(num_parents):
        max_fitness_idx = np.where(loFiteness == np.max(loFiteness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        loFiteness[max_fitness_idx] = -99999999999
    return parents


def select_mating_pool(pop, fitness, num_parents, maximize=True):
    """
    Modified to make it parameterizable
    """
    # Selecting the best individuals in the current generation as parents
    # for producing the offspring of the next generation.

    if maximize is True:
        reset_value = -1e11
    elif maximize is False:
        reset_value = 1e11
    parents = np.empty((num_parents, pop.shape[1]))
    # Need to create a copy otherwise the global variable gets affected.
    loFiteness = fitness.copy()

    for parent_num in range(num_parents):
        if maximize is True:
            extreme_fitness_idx = np.where(
                    loFiteness == np.max(loFiteness))
        elif maximize is False:
            extreme_fitness_idx = np.where(
                    loFiteness == np.min(loFiteness))
        extreme_fitness_idx = extreme_fitness_idx[0][0]
        parents[parent_num, :] = pop[extreme_fitness_idx, :]
        loFiteness[extreme_fitness_idx] = reset_value
    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents.
    # Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1) % parents.shape[0]
        # The new offspring will have its first half of
        # its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[
                parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of
        # its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[
                parent2_idx, crossover_point:]
    return offspring


def mutation_org(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(
            offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the
    # num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[
                    idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover


def mutation(offspring_crossover, num_mutations=1):

    # Mutation changes a number of genes as defined by the
    # num_mutations argument. The changes are random.

    for idx in range(offspring_crossover.shape[0]):  # For every offspring
        gene_ids = np.random.choice(offspring_crossover.shape[1],
                                    num_mutations, replace=False)

        for i in range(num_mutations):
            offspring_crossover[idx, gene_ids[i]] = np.random.uniform(
                    low=-4.0, high=4.0)

    return offspring_crossover
