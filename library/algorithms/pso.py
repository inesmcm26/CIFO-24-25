import random
import numpy as np
from copy import deepcopy
from solution import PSOSolution

def particle_swarm_optimization(
    population: list[PSOSolution],
    w=0.5,          # inertia weight
    c1=1.5,         # cognitive coefficient
    c2=1.5,         # social coefficient
    maximization=False,
    max_iter=100,
    verbose=False
):
    # 1. Initialize particles and velocities
    # and
    # 2. Set personal bests
    # are already done in the initial population, as it consists of PSOSolutions

    # 3. Set global best
    if maximization:
        global_best = deepcopy(max(population, key=lambda p: p.fitness()))
    else:
        global_best = deepcopy(min(population, key=lambda p: p.fitness()))

    global_best_history = []

    # 2. Repeat until termination condition
    for iter in range(max_iter):
        if verbose:
            print(f"Iteration: {iter+1}")
        # 2.1. For each particle
        for particle in population:
            if verbose:
                print(f"Initial particle location: {particle.repr}")
            
            # Update particle position
            new_position = particle.repr + particle.velocity

            # Update particle velocity
            r1 = np.array([random.random() for _ in range(len(particle.repr))])
            r2 = np.array([random.random() for _ in range(len(particle.repr))])

            inertia = w * particle.velocity
            cognitive = np.multiply(c1 * r1 , particle.best_repr - particle.repr)
            social = np.multiply(c2 * r2, global_best.repr - particle.repr)

            particle.velocity = inertia + cognitive + social

            particle.repr = new_position

            # Update personal best
            if particle.fitness() < particle.best_fitness:
                particle.best_repr = deepcopy(particle.repr)
                particle.best_fitness = particle.fitness()

            if verbose:
                print(f"New particle location: {particle.repr}")
                print(f"Updated velocity: {particle.velocity}")

        # Update global best
        if maximization:
            global_best_iter = deepcopy(max(population, key=lambda p: p.fitness()))
            if global_best_iter.fitness() > global_best.fitness():
                global_best = global_best_iter
        else:
            global_best_iter = deepcopy(min(population, key=lambda p: p.fitness()))
            if global_best_iter.fitness() < global_best.fitness():
                global_best = deepcopy(global_best_iter)

        global_best_history.append(global_best)

        if verbose:
            print(f"Best Fitness: {global_best.fitness()}")
            print("---------------------")

    return global_best, global_best_history
