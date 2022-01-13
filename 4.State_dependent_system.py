'''
In this example the Deep Symbolic Regression is used to reconstruct the following: "[x[1], -k*x[0] - c*x[1] - F0*np.sin(omega*t)*x[1]]". The full system is reconstructed, making use of the observed acceleration. Here the library of functions is a combination of a-priori defined ones and others resulting form the Genetic Programming-based Symbolic Regression.

Author(s): Matteo Manzi
email: matteo.manzi@strath.ac.uk

13-10-2020
'''

import operator
import numpy as np

from scoop import futures
from deap import base, creator, gp, tools
import random

from scipy.integrate import solve_ivp
from scipy.integrate import simps
from scipy import optimize

import pysindy as ps
from utils import *

import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib as mpl

seed = 550
random.seed(seed)
mpl.use('Agg')


def my_varAnd(population, toolbox_local, cxpb, mutpb):
    offspring = [toolbox_local.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    # for h_component in range(ntrees):
    for i in range(1, len(offspring), 2):
        # for h_component in range(ntrees):
        if random.random() < cxpb:
            h_component = random.randint(0, ntrees-1)
            offspring[i - 1][h_component], offspring[i][h_component] = toolbox_local.mate(offspring[i - 1][h_component], offspring[i][h_component])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        for h_component in range(ntrees):
            if random.random() < mutpb:
                # h_component = random.randint(0, ntrees-1)
                offspring[i][h_component], = toolbox_local.mutate(offspring[i][h_component])
                del offspring[i].fitness.values

    # for i in range(len(offspring)):
    #     for j in range(len(offspring[0])):
    #         print(offspring[i][j])

    return offspring


def my_eaSimple(population, toolbox_local, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox_local.map(toolbox_local.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox_local.select(population, len(population))

        # Vary the pool of individuals
        offspring = my_varAnd(offspring, toolbox_local, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox_local.map(toolbox_local.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
            for i in range(ntrees):
                print(halloffame[0][i])

    return population, logbook


def dxdt(t, x):
    dxdt_vector = [x[1], -k*x[0] - c*x[1] - F0*np.sin(omega*t)*x[1]]
    return dxdt_vector


def observedstate(t, i):
    state_history = solve_ivp(dxdt, y0=x0_nominal, t_span=[t_start, t], rtol=reltol, atol=abstol).y

    final_state = state_history[i, -1]
    return final_state


# Initialize PYSINDY:
poly_library = ps.PolynomialLibrary(degree=1)
# stlsq_optimizer = ps.STLSQ(threshold=.01, alpha=.5)

model = ps.SINDy() # optimizer=stlsq_optimizer

# ###################################    P R I M I T I V E  -  S E T     ###################################
ntrees = 5  # Number of trees defining an individual
nc = 1  # Number of symbolic constants associated to the individual.

dimensions = 1  # What's the size of the velocity vector?
size_input = dimensions*2 + 1 + nc   # 1 + nc  #
intypes = []
for i in range(size_input):
    intypes.append(float)

pset = gp.PrimitiveSetTyped("MAIN", intypes, float)

pset.addPrimitive(np.multiply, [float, float], float, name="mul")
pset.addPrimitive(np.sin, [float], float, name="sin")
# pset.addPrimitive(np.cos, [float], float, name="cos")
# pset.addPrimitive(np.square, [float], float, name="square")

# objects:
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='v')
# # ... add time as an independent variable, if necessary
pset.renameArguments(ARG2='t')

pset.renameArguments(ARG3='p1')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Subindividual", gp.PrimitiveTree)
creator.create("Individual", list, fitness=creator.FitnessMin)

#  ####################################### T O O L B O X #########################################
toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, type_=pset.ret, min_=1, max_=3)
toolbox.register("subindividual", tools.initIterate, creator.Subindividual, toolbox.expr)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.subindividual, n=ntrees)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("select", tools.selTournament, tournsize=2)


def random_mating_operator(ind1, ind2):
    roll = random.random()
    if roll < 0.5:
        return gp.cxOnePoint(ind1, ind2)
    elif roll < 1.5:
        return gp.cxOnePointLeafBiased(ind1, ind2, termpb=0.5)


def random_mutation_operator(individual):
    roll = random.random()
    if roll < 0.5:
        return gp.mutInsert(individual, pset=pset)
    elif roll < 0.66:
        return gp.mutShrink(individual)
    elif roll < 2.66:
        return gp.mutNodeReplacement(individual, pset=pset)


toolbox.register("mate", random_mating_operator)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=1)
toolbox.register("mutate", random_mutation_operator)

k = 4.518
c = 0.376
F0 = 2.865
omega = 1.447

x0 = 2
x0dot = 3
x0_nominal = [x0, x0dot]  # State vector here
posvelacc0_measured = [x0, x0dot, dxdt(0, [x0, x0dot])[1]]

reltol = 1e-4
abstol = 1e-4
delta_t = 0.1
t_start = 0.0
t_end = 25

t_true = np.arange(t_start, t_end*2.0, delta_t)
x_true = solve_ivp(dxdt, y0=x0_nominal, t_span=[t_true[0], t_true[-1]], t_eval=t_true, rtol=reltol, atol=abstol).y
acc_true = dxdt(t_true, x_true)[1]
posvelacc_true = np.vstack([x_true, acc_true])

time_rec_obs = [time for time in [0.1, 0.3, 0.8, 1, 1.7, 3, 4, 4.3, 5, 6, 7, 8, 8.4, 9.5, 10.8, 11, 11.5, 13, 15, 18, 19, 19.3, 19.8, 20, 21.3, 22, 24]]
half_dt_obs = 0.01
time_obs = []
for i in range(len(time_rec_obs)):
    time_obs.append(time_rec_obs[i] - half_dt_obs)
    time_obs.append(time_rec_obs[i] + half_dt_obs)


position_obs = [observedstate(t, 0) for t in time_obs]
velocity_obs = [observedstate(t, 1) for t in time_obs]

# For the state, now interpolate:
position_rec_obs = []
for i in range(len(time_rec_obs)):
    position_rec_obs.append((position_obs[2*i]+position_obs[2*i+1])/2.0)

velocity_rec_obs = []
for i in range(len(time_rec_obs)):
    velocity_rec_obs.append((velocity_obs[2 * i] + velocity_obs[2 * i + 1]) / 2.0)

# For the acceleration, for now finite difference:
acc_rec_obs = []
for i in range(len(time_rec_obs)):
    acc_rec_obs.append((velocity_obs[2*i+1]-velocity_obs[2*i])/(2.0*half_dt_obs))

x_train = [position_rec_obs, velocity_rec_obs, time_rec_obs]
x_train = np.asarray(x_train).transpose()

x_dot_train = [velocity_rec_obs, acc_rec_obs, np.ones(len(time_rec_obs)).tolist()]
x_dot_train = np.asarray(x_dot_train).transpose()


# #################################  F I T N E S S    F U N C T I O N    ###################
def evalSymbReg(individual, flag_solution=0):
    print('')
    # print('Individual:')
    for i in range(ntrees):
        print(individual[i])

    # Transform the tree expression in a callable function
    sr_functions = []
    for i in range(ntrees):
        sr_functions.append(toolbox.compile(expr=individual[i]))

    def coeff_estimation(p):
        # ############################ P Y S I N D Y #################################
        # BUG: the lambda function is overwriting inside the for loop.
        local_sr_functions = []
        for i in range(ntrees):
            fn = lambda x, v, t: sr_functions[i](x, v, t, *p)  #   v*np.sin(p*t)   #
            local_sr_functions.append(fn)

        custom_library = ps.CustomLibrary(library_functions=local_sr_functions)
        library = custom_library + poly_library
        model = ps.SINDy(feature_library=library)
        model.fit(x_train, t=np.array(time_rec_obs), x_dot=x_dot_train)
        # fitness = - model.score(x_train, t=np.array(time_rec_obs), x_dot=x_dot_train)

        x_eval_sim = model.simulate(x_train[0], time_rec_obs, rtol=reltol, atol=abstol, hmin=1e-6)  #
        fitness = simps(np.square(x_eval_sim.transpose()[0]-position_rec_obs), x=time_rec_obs) # simps(np.square((x_eval_sim.transpose()[0]-position_rec_obs)/np.max(position_rec_obs)) + np.square((x_eval_sim.transpose()[1]-velocity_rec_obs)/np.max(velocity_rec_obs)), x=time_rec_obs)

        # fitness = np.abs(simps(x_eval_sim.transpose()[0]-position_rec_obs, x=time_rec_obs)) - model.score(x_train, t=np.array(time_rec_obs), x_dot=x_dot_train)

        # model.print()
        # print('Local fitness:')
        # print(fitness)
        # print(' ')

        return fitness
    #
    # p_landscape = np.arange(omega - 1.0, omega + 1.0, 0.01).tolist()
    # # test = p_landscape[4]
    # # test2 = coeff_estimation([3.2])
    # fitness_landscape = []
    # for i in range(len(p_landscape)):
    #     fitness_landscape.append(coeff_estimation([p_landscape[i]]))
    #
    # plt.figure()
    # plt.plot(p_landscape, fitness_landscape)
    # plt.savefig('figures/landscape.png', dpi=600)

    if flag_solution == 0:
        OptResults = optimize.minimize(coeff_estimation, 1.447, tol=1e-1)  # random.random()*1.5   # ,  random.random()*10  #  options={"maxiter": 2}
    else:
        OptResults = optimize.differential_evolution(coeff_estimation, [(0, 10)], maxiter=100, seed=seed, disp=True)
        # OptResults = optimize.minimize(coeff_estimation, 1.4, options={"maxiter": 10})  # random.random()*10   #  tol=1e-2,
        #   optimize.basinhopping(coeff_estimation, random.random()*10, seed=seed, disp=True)
        # OptResults = optimize.minimize(coeff_estimation, random.random()*10)

    k_array = OptResults.x
    fitness = OptResults.fun
    # fitness = 1000000

    # model.print()

    # print('Constants:')
    # print(k_array)
    # print('Fitness:')
    # print(fitness)
    # print(' ')

    if flag_solution == 1:
        print('Constants:')
        print(k_array)
        np.savetxt('output/k_array.txt', k_array, delimiter=',')

        print('Fitness:')
        print(fitness)

        final_library_functions = []
        for i in range(ntrees):
            fn = lambda x, v, t: sr_functions[i](x, v, t, *k_array)  # x[0], x[1], x[2]
            final_library_functions.append(fn)
        custom_library = ps.CustomLibrary(library_functions=final_library_functions)
        library = custom_library + poly_library

        model = ps.SINDy(feature_library=library)
        model.fit(x_train, t=np.array(time_rec_obs), x_dot=x_dot_train)

        model.print()

        clims = [-30, 30]
        # cmap = mpl.GnBu
        plt.figure()
        plt.imshow(sigma_plot_sparsity(model.coefficients().T), interpolation='nearest', cmap='GnBu')
        plt.clim(clims)
        # plt.axis('off')
        plt.savefig('figures/coefficients_naAlpha.png', dpi=600)

        x_ntest = solve_ivp(dxdt, y0=x0_nominal, t_span=[t_true[0], t_true[-1]], t_eval=t_true, rtol=reltol,
                           atol=abstol).y

        x0_nominal.append(t_start)
        x0_ntest = np.asarray(x0_nominal)

        # Evolve the initial condition in time with the SINDy model
        x_ntest_sim = model.simulate(x0_ntest, t_true)

        fig, axs = plt.subplots(x_ntest.shape[0], 1, sharex=True, figsize=(7, 9))
        for i in range(x_ntest.shape[0]):
            axs[i].plot(t_true, x_ntest_sim[:, i], label='model simulation', linewidth=2)  #  'r--',
            axs[i].plot(t_true, x_ntest[i, :], color='#888888', linestyle = '--', label='true simulation', linewidth=2)  # 'k'

            axs[i].scatter(0, posvelacc0_measured[i], color='red', label='Observations')
            for j in range(len(time_rec_obs)):
                axs[i].scatter(time_rec_obs[j], x_train[j][i], color='red')

            axs[i].legend()
            axs[i].set(xlabel=xlabel, ylabel=ylabels[i])
            axs[i].grid(True)
            plt.tight_layout()
        plt.savefig('figures/solution_x.png')

        # xv0_test = [1.5, 4.12]
        # xv0_test.append(t_start)
        # x0_test = x0_nominal  #  np.asarray(xv0_test)
        # print(x0_test)
        #
        # t_test = np.arange(t_start, t_end*2, delta_t)
        # x_test = solve_ivp(dxdt, y0=x0_test, t_span=[t_test[0], t_test[-1]], t_eval=t_test, rtol=reltol,
        #                     atol=abstol).y
        #
        # # Evolve the NEW initial condition in time with the SINDy model
        # x_test_sim = model.simulate(x0_test, t_true)
        #
        # fig, axs = plt.subplots(x_test.shape[0], 1, sharex=True, figsize=(7, 9))
        # for i in range(x_test.shape[0]):
        #     axs[i].plot(t_true, x_test_sim[:, i], label='model simulation', linewidth=2)  # 'r--',
        #     axs[i].plot(t_true, x_test[i, :], color='#888888', linestyle='--', label='true simulation',
        #                 linewidth=2)  # 'k'
        #
        #     axs[i].legend()
        #     axs[i].set(xlabel=xlabel, ylabel=ylabels[i])
        #     axs[i].grid(True)
        #     plt.tight_layout()
        # plt.savefig('figures/newIC.png')

    return [fitness, ]


toolbox.register("evaluate", evalSymbReg)
toolbox.register("map", futures.map)

history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))

xlabel = r"$t \ [s]$"
ylabels = [r"$x \ [m]$", r"$\dot{x} \ [m/s]$", r"$\ddot{x} \ [m/s^2]$"]


# ################################ M A I N ###############################################
def main():
    x_xdot_xddot = [position_rec_obs, velocity_rec_obs, acc_rec_obs]

    # Plot the true solution & Observations
    fig, axs = plt.subplots(dimensions*3, 1, sharex=True, figsize=(7, 9))
    # mpl.rcParams['legend.fontsize'] = 22
    # plt.rcParams.update({'font.size': 22})  # increase the font size
    for i in range(dimensions*3):
        axs[i].plot(t_true, posvelacc_true[i, :], color='#888888', label='True Solution', linewidth=2)
        axs[i].scatter(0, posvelacc0_measured[i], color='red', label='Observations')
        for j in range(len(time_rec_obs)):
            axs[i].scatter(time_rec_obs[j], x_xdot_xddot[i][j], color='red')
        axs[i].legend()
        axs[i].set(xlabel=xlabel, ylabel=ylabels[i])
        axs[i].grid(True)
        plt.tight_layout()
    plt.savefig('figures/xv_nominal.png')

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=15)  # 10
    hof = tools.HallOfFame(1)
    # EVOLVE HERE:
    pop, log = my_eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.8, ngen=20, stats=mstats, halloffame=hof, verbose=True)

    # Plot trees of differential equation
    expr = hof[0]
    for i in range(ntrees):
        tree = plt.figure()
        nodes, edges, labels = gp.graph(expr[i])
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = graphviz_layout(g, prog="dot")

        nx.draw_networkx_nodes(g, pos, node_size=5000, node_color='b')
        nx.draw_networkx_edges(g, pos, width=2.0, edge_color='k')
        nx.draw_networkx_labels(g, pos, labels, font_size=20.0, font_color='w')

        plt.axis('off')
        plt.margins(0.2)
        # plt.tight_layout()
        plt.savefig('figures/tree%s.png' % i)

    # gen_tree = plt.figure()
    # graph = nx.DiGraph(history.genealogy_tree)
    # graph = graph.reverse()  # Make the graph top-down
    # colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
    # nx.draw(graph, node_color=colors)
    # plt.savefig('figures/genealogy_tree.png')

    print('Best individual : ')
    for i in range(ntrees):
        print(hof[0][i])
    # print('Fitness: ', hof[0].fitness)
    evalSymbReg(hof[0], flag_solution=1)

    return pop, log, hof


if __name__ == "__main__":
    main()
