import random
from scipy import special as sc
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas
from matplotlib.table import Table


def fitness_score(seq):
    score = 0

    for row in range(NUM_QUEENS):
        col = seq[row]

        for other_row in range(NUM_QUEENS):

            # queens cannot pair with itself
            if other_row == row:
                continue
            if seq[other_row] == col:
                continue
            if other_row + seq[other_row] == row + col:
                continue
            if other_row - seq[other_row] == row - col:
                continue
            # score++ if every pair of queens are non-attacking.
            score += 1

    # divide by 2 as pairs of queens are commutative
    return score / 2


def selection(populate):
    parents = []
    for ind in populate:
        # select parents with probability proportional to their fitness score
        if random.randrange(sc.comb(NUM_QUEENS, 2) * 2) < fitness_score(ind):
            parents.append(ind)

    return parents


def crossover(parents):
    # random indexes to to cross states with
    cross_points = random.sample(range(NUM_QUEENS), MIXING_NUMBER - 1)
    offsprings = []

    # all permutations of parents
    permutations = list(itertools.permutations(parents, MIXING_NUMBER))

    for perm in permutations:
        offspring = []

        # track starting index of sublist
        start_pt = 0

        for parent_idx, cross_point in enumerate(cross_points):  # doesn't account for last parent

            # sublist of parent to be crossed
            parent_part = perm[parent_idx][start_pt:cross_point]
            offspring.append(parent_part)

            # update index pointer
            start_pt = cross_point

        # last parent
        last_parent = perm[-1]
        parent_part = last_parent[cross_point:]
        offspring.append(parent_part)

        # flatten the list since append works kinda differently
        offsprings.append(list(itertools.chain(*offspring)))

    return offsprings


def mutate(seq):
    for row in range(len(seq)):
        if random.random() < MUTATION_RATE:
            seq[row] = random.randrange(NUM_QUEENS)

    return seq


def print_found_goal(populations, to_print=True):
    global solution
    for ind in populations:
        score = fitness_score(ind)
        print(score)
        if score == sc.comb(NUM_QUEENS, 2):
            if to_print:
                print('Solution found')
                solution = ind
            return True
        break

    if to_print:
        print('Solution not found')
    return False


def evolution(populate):
    # select individuals to become parents
    parents = selection(populate)

    # recombination. Create new offsprings
    offsprings = crossover(parents)

    # mutation
    offsprings = list(map(mutate, offsprings))

    # introduce top-scoring individuals from previous generation and keep top fitness individuals
    new_gen = offsprings

    for ind in populate:
        new_gen.append(ind)

    new_gen = sorted(new_gen, key=lambda i: fitness_score(i), reverse=True)[:POPULATION_SIZE]

    return new_gen


def generate_population():
    initPopulation = []

    for individual in range(POPULATION_SIZE):
        new = [random.randrange(NUM_QUEENS) for _ in range(NUM_QUEENS)]
        initPopulation.append(new)

    return initPopulation


# Constants, experiment parameters
NUM_QUEENS = 100
POPULATION_SIZE = 15
MIXING_NUMBER = 2 
MUTATION_RATE = 0.0199

# Running the experiment
generation = 0

# generate random population
population = generate_population()
solution = []
lables = []

for i in range(NUM_QUEENS):
    lables.append(i)

while not print_found_goal(population):
    print(f'Generation: {generation}')
    print_found_goal(population)
    population = evolution(population)
    generation += 1

print(solution)

def graph():
    data = pandas.DataFrame(np.random.random((NUM_QUEENS, NUM_QUEENS)))
    checkerboard_table(data)
    plt.show()

def checkerboard_table(data, bkg_colors=None):
    if bkg_colors is None:
        bkg_colors = ['gray', 'white']

    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = data.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    for i in range(NUM_QUEENS):
        for j in range(NUM_QUEENS):
            idx = [j % 2, (j + 1) % 2][i % 2]
            color = bkg_colors[idx]
            if solution[i] == j:
                tb.add_cell(i, j, width, height,
                            loc='center', facecolor='black')
            else:
                tb.add_cell(i, j, width, height,
                            loc='center', facecolor=color)

    # Row Labels...
    for i, label in enumerate(data.index):
        tb.add_cell(i, -1, width, height, text=label, loc='right',
                    edgecolor='none', facecolor='none')
    # Column Labels...
    for j, label in enumerate(data.columns):
        tb.add_cell(-1, j, width, height / 2, text=label, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)
    return fig


graph()
