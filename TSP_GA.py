import random
import time

import tsplib95


def selectProblem(problem):
    """
    Selecting a problem from TSP problem database
    Input:
    1- Problem name
    Output:
    String containing the problem from TSP problem database
    """
    with open('ALL_tsp/' + str(problem) + '.tsp') as f:
        text = f.read()
    return tsplib95.parse(text)


def dist_two_cities(city_1, city_2, problem):
    """
    Calculating the distance between two cities
    Input:
    1- City one name
    2- City two name
    3- TSP problem
    Output:
    Calculated Euclidean distance between two cities
    """
    edge = city_1, city_2
    return problem.get_weight(*edge)


def total_dist_individual(individual, problem):
    """
    Calculating the total distance traveled by individual,
    one individual means one possible solution (1 permutation)
    Input:
    1- Individual list of cities
    Output:
    Total distance traveled
    """

    total_dist = 0
    for i in range(0, len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_cities(individual[i], individual[0], problem)
        else:
            total_dist += dist_two_cities(individual[i], individual[i + 1], problem)
    return total_dist


def initial_population(cities_list, problem, n_population=250):
    """
    Generating initial population of cities
    randomly shuffled list of cities
    Input:
    1- Cities list
    2. Number of population
    Output:
    Generated lists of cities
    """
    population_perms = []
    while len(population_perms) < n_population:
        random.shuffle(cities_list)
        population_perms.append([total_dist_individual(cities_list, problem), cities_list.copy()])
    return population_perms


def fitness_prob(population):
    """
    Calculating the fitness probability
    Input:
    1- Population
    Output:
    Population fitness probability
    """
    total_dist_all_individuals = []

    for i in range(0, len(population)):
        total_dist_all_individuals.append(population[i][0])

    max_population_cost = max(total_dist_all_individuals)
    population_fitness = [max_population_cost - x for x in total_dist_all_individuals]
    # population_fitness = [1/x for x in total_dist_all_individuals]
    population_fitness_sum = sum(population_fitness)
    population_fitness_probs = [x / population_fitness_sum for x in population_fitness]

    return population_fitness_probs


def roulette_wheel(population):
    """
    Implement a selection strategy based on proportionate roulette wheel
    Selection.
    Input:
    1- population
    2: fitness probabilities
    Output:
    selected individual
    """
    cumulative_probabilities = []
    cumulative_probability = 0.0

    fitness_probs = fitness_prob(population)

    for prob in fitness_probs:
        cumulative_probability += prob
        cumulative_probabilities.append(cumulative_probability)

    r = random.random()
    for i, cum_prob in enumerate(cumulative_probabilities):
        if r <= cum_prob:
            selected = population[i]
            break

    return selected


def crossover(parent_1, parent_2, problem):
    """
    Implement mating strategy using simple crossover between two parents
    Input:
    1- parent 1
    2- parent 2
    Output:
    1- offspring 1
    2- offspring 2
    """
    n_cities_cut = problem.dimension - 1
    cut = round(random.uniform(1, n_cities_cut))
    parent_1 = parent_1[1]
    parent_2 = parent_2[1]

    offspring_1 = parent_1[0:cut]
    offspring_1 += [city for city in parent_2 if city not in offspring_1]

    offspring_2 = parent_2[0:cut]
    offspring_2 += [city for city in parent_1 if city not in offspring_2]

    return offspring_1, offspring_2


def mutation(offspring, mutation_per, problem):
    """
    Implement mutation strategy in a single offspring
    Input:
    1- offspring individual
    Output:
    1- mutated offspring individual
    """
    n_cities_cut = problem.dimension - 1
    mutate_threshold = random.random()
    if mutate_threshold > (1 - mutation_per):
        index_1 = round(random.uniform(0, n_cities_cut))
        index_2 = round(random.uniform(0, n_cities_cut))

        temp = offspring[index_1]
        offspring[index_1] = offspring[index_2]
        offspring[index_2] = temp
    return offspring


def tournament_selection(population, fitness_values, tournament_size, problem):
    """
    Implement a tournament selection strategy
    Input:
    1- population
    2- fitness probabilities
    3- tournament size
    Output:
    Selected individual
    """
    selected = []
    population_size = len(population)

    # Repeat the tournament selection process until we've selected enough individuals
    while len(selected) < population_size:
        tournament_members = random.sample(range(population_size), tournament_size)

        # Find the individual with the highest fitness within the tournament
        winner_index = max(tournament_members, key=lambda x: fitness_values[x])

        # Add the winner to the selected individuals
        selected.append(population[winner_index])

    return selected


def findBestOffspring(population):
    """
    Finds the best offspring from given population
    Input:
    1- population
    Output:
    Best offspring
    """
    min = float('inf')
    min_index = None
    for i in range(len(population)):
        if population[i][0] < min:
            min = population[i][0]
            min_index = i
    return population[min_index]


def solution(populationStart, n_generations, mutation_per):
    """
    Genetic algorithm that solves the TSP
    Input:
    1- initial population
    2- number of generations GA will have
    3- mutation rate
    Output:
    Best offspring of last (best) generation
    """
    population = initial_population(populationStart, problem, n_population=200)
    old_best = findBestOffspring(population)

    for i in range(n_generations):
        next_population = []
        next_best = findBestOffspring(population)
        next_population.append(next_best)

        while len(next_population) < len(population):
            parents = {0: roulette_wheel(population),
                       1: roulette_wheel(population)}

            # roditelje maknut
            next_population.append(parents[0])
            next_population.append(parents[1])

            child_1, child_2 = crossover(parents[0], parents[1], problem)
            child_1 = mutation(child_1, mutation_per, problem)
            child_2 = mutation(child_2, mutation_per, problem)

            next_population.append([total_dist_individual(child_1, problem), child_1])
            next_population.append([total_dist_individual(child_2, problem), child_2])

        population = next_population[:200]
        if old_best != next_best:
            old_best = next_best
            print("Generation " + str(i))
            print("Best Individual: " + str(next_best[1]))
            print("Best Individual Cost = " + str(next_best[0]))

    return findBestOffspring(population)


# 2085
# problem = selectProblem("gr17")
# 1610
problem = selectProblem("bayg29")
# 25395 (najbolji) 5000 iteracija 0.8 -> 29369
# problem = selectProblem("brazil58")
# 937 (najbolji) 5000 iteracija 0.8 -> 962
# problem = selectProblem("fri26")

cities = list(problem.get_nodes())
time_start = time.time()
best = solution(cities, 3000, 0.6)
time_end = time.time()

time_cost = time_end - time_start
print("Time cost: " + str(time_cost))
# print(best)
# print(total_dist_individual(best[1], problem))
