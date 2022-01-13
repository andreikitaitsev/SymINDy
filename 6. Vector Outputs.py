# # Two Body Problem with Unknown Drag - Vector Output

def warn(*args, **kwargs):  # removes all wanrings from the script
    pass
import warnings
warnings.warn = warn


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

model = ps.SINDy()
ps.STLSQ(normalize=True, alpha=0.1)
seed = 550
random.seed(seed)
mpl.use('Agg')


# # Evolutionary Algorithm


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


# # Crossover and Mutation Operations Function


def my_varAnd(population, toolbox_local, cxpb, mutpb):
    offspring = [toolbox_local.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
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
    return offspring


var_list = ["r", "theta", "vr", "vt", "t"]  # list of strings for each component


# ### Simulate True Data

def dxdt_true(t, x):
    return [
        x[2],

        x[3] / x[0],

        -mu / (x[0] ** 2) + (x[3] ** 2) / x[0] - 0.5 * A * (1 + 0.5 * np.sin(omega * t)) * (
                    1 / m) * rho * Cd * np.linalg.norm([x[2], x[3]]) * x[2],

        -x[2] * x[3] / x[0] - 0.5 * A * (1 + 0.5 * np.sin(omega * t)) * (1 / m) * rho * Cd * np.linalg.norm(
            [x[2], x[3]]) * x[3],

    ]


def dxdt_true(t, x):
    return [
        x[2],

        x[3] / x[0],

        -mu / (x[0] ** 2) + (x[3] ** 2) / x[0] - A * (1 / m) * rho * Cd * np.linalg.norm([x[2], x[3]]) * x[2],

        -x[2] * x[3] / x[0] - A * (1 / m) * rho * Cd * np.linalg.norm([x[2], x[3]]) * x[3],

    ]

def dxdt_expected(t, x):
    return [x[2], x[3]/x[0], -mu/(x[0]**2) + (x[3]**2)/x[0], -x[2]*x[3]/x[0]]


mu = 3.986004418e+14 # Standard Gravitational Parameter of Earth, m^3s^-2
omega = 2*np.pi/60   # Non-Linear Parameter, rad
A = 0.1              # Average Effective Area, m^2
m = 1                # Mass, kg
Cd = 2.2             # Ballistic Coefficient

r0 = 6521e+03        # Initial Altitude, m
theta0 = 0           # Initial True Anomaly, rad
vr0 = 0.05e+03       # Initial Radial Velocity, m/s
vt0 = np.sqrt(mu/r0) # Initial Tangential Velocity, m/s

rho = 1.225 * np.exp(-(r0-6371e+03)*9.81*0.0289644/8.3144598/273.15)


t_start = 0
t_end = 1200
delta_t = 10
t_true = np.arange(t_start, t_end*2, delta_t)


x0_nominal = [r0, theta0, vr0, vt0]

posvelacc0_measured = [r0, theta0, vr0, vt0, dxdt_true(0, x0_nominal)[2], dxdt_true(0, x0_nominal)[3]]


reltol = 1e-3; abstol = 1e-3

x_true = solve_ivp(dxdt_true, y0=x0_nominal, t_span=[t_true[0], t_true[-1]], t_eval=t_true, rtol=reltol, atol=abstol).y

acc_true = np.vstack([dxdt_true(t_true, x_true)[2], dxdt_true(t_true, x_true)[3]])

posvelacc_true = np.vstack([x_true, acc_true])


# ## Simulate Stochastic Observations


observations = 100  # how many observations?

time_rec_obs = t_end * np.random.rand(observations)    # Generates an array of shape (observations,)
time_rec_obs = np.sort(time_rec_obs).tolist()  # Sorts the array chronologically and convert from np array to a list

half_dt_obs = delta_t/10
time_obs = []
for i in range(len(time_rec_obs)):
    time_obs.append(time_rec_obs[i] - half_dt_obs)
    time_obs.append(time_rec_obs[i] + half_dt_obs)


def observedstate(t, i):
    state_history = solve_ivp(dxdt_true, y0=x0_nominal, t_span=[t_start, t], rtol=reltol, atol=abstol).y

    final_state = state_history[i, -1]
    return final_state


r_obs = [observedstate(t, 0) for t in time_obs]
theta_obs = [observedstate(t, 1) for t in time_obs]
vr_obs = [observedstate(t, 2) for t in time_obs]
vt_obs = [observedstate(t, 3) for t in time_obs]


r_rec_obs = []
for i in range(len(time_rec_obs)):
    r_rec_obs.append((r_obs[2*i]+r_obs[2*i+1])/2.0)

theta_rec_obs = []
for i in range(len(time_rec_obs)):
    theta_rec_obs.append((theta_obs[2 * i] + theta_obs[2 * i + 1]) / 2.0)
    
vr_rec_obs = []
for i in range(len(time_rec_obs)):
    vr_rec_obs.append((vr_obs[2 * i] + vr_obs[2 * i + 1]) / 2.0)
    
vt_rec_obs = []
for i in range(len(time_rec_obs)):
    vt_rec_obs.append((vt_obs[2 * i] + vt_obs[2 * i + 1]) / 2.0)
    
ar_rec_obs = []
for i in range(len(time_rec_obs)):
    ar_rec_obs.append((vr_obs[2*i+1]-vr_obs[2*i])/(2.0*half_dt_obs))
    
at_rec_obs = []
for i in range(len(time_rec_obs)):
    at_rec_obs.append((vt_obs[2*i+1]-vt_obs[2*i])/(2.0*half_dt_obs))


x_train = [r_rec_obs, theta_rec_obs, vr_rec_obs, vt_rec_obs, time_rec_obs]
x_train = np.asarray(x_train).transpose()  # x_train.shape = (30,5)

x_dot_expected = np.array([dxdt_expected(time_rec_obs[i], [r_rec_obs[i], theta_rec_obs[i], vr_rec_obs[i], vt_rec_obs[i]]) for i in range(len(time_rec_obs))])  # x_dot_expected.shape = (30,4)

x_dot_obs = [vr_rec_obs, np.array(vt_rec_obs)/np.array(r_rec_obs), ar_rec_obs, at_rec_obs]
x_dot_obs = np.asarray(x_dot_obs).transpose()

dx_dot_train = [np.zeros(len(time_rec_obs)), np.zeros(len(time_rec_obs)), np.array(ar_rec_obs)-x_dot_expected[:, 2], np.array(at_rec_obs)-x_dot_expected[:, 3], np.zeros(len(time_rec_obs))]
dx_dot_train = np.asarray(dx_dot_train).transpose()  # dx_dot_train.shape = (30,5)


### PRIMITIVE SET ###
ntrees = 2  # Number of trees defining an individual
nc = 1  # Number of symbolic constants associated to the individual.

dimensions = 2  # Size of the velocity vector, [vr,vt]'

in_vectors = 2
in_scalars = 1 + nc


# Types Associated to an Individual are the two vectors and 2 scalars.
intypes = []
for i in range(in_vectors):
    intypes.append(np.ndarray)

for i in range(in_scalars):
    intypes.append(float)


def norm_fun(vector):
    return np.sqrt(vector[0]*vector[0]+vector[1]*vector[1])


pset = gp.PrimitiveSetTyped("MAIN", intypes, np.ndarray)

pset.addPrimitive(np.multiply, [float, np.ndarray], np.ndarray, name="vmul")
pset.addPrimitive(np.multiply, [float, float], float, name="mul")
pset.addPrimitive(np.sin, [float], float, name="sin")
pset.addPrimitive(norm_fun, [np.ndarray], float, name="norm")


# objects:
pset.renameArguments(ARG0='X_vec')
pset.renameArguments(ARG1='V_vec')
pset.renameArguments(ARG2='t')
pset.renameArguments(ARG3='p1')


# The classes are then initialized and registered in the toolbox


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Subindividual", gp.PrimitiveTree)
creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genGrow, pset=pset, type_=pset.ret, min_=1, max_=2)
toolbox.register("subindividual", tools.initIterate, creator.Subindividual, toolbox.expr)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.subindividual, n=ntrees)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

toolbox.register("select", tools.selTournament, tournsize=2)


# Mutation and Crossover Functions are Registered in the Toolbox to Be Used in the Variation Part of the Algorithm


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
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=2)
toolbox.register("mutate", random_mutation_operator)


# Multiply Derivatives by a Constant to Increase Overall Magnitude
# note: SINDy linear parameter will need to be divided by this value at the end
mag_factor = 1e3

dx_dot_train = mag_factor * dx_dot_train


# ## Symbolic Regression

def evalSymbReg(individual, flag_solution=0):

    def coeff_estimation(p):
        ################################# P Y S I N D Y #################################
        def make_SINDy_func(tree, dimension):
            # function which converts a deap tree object to a lambda function for pySINDy
            sr_func = lambda r, theta, vr, vt, t: toolbox.compile(tree)([r, theta], [vr, vt], t, *p)[dimension]
            return sr_func
        
        funcs = []
        for i in range(ntrees):
            for j in range(dimensions):
                funcs.append(make_SINDy_func(individual[i], j))
                # generate a list of lambda functions corresponding to GP trees. len(funcs)dimensions*ntrees
            
        library = ps.CustomLibrary(library_functions=funcs)
        
        model = ps.SINDy(optimizer=ps.STLSQ(threshold=1e-10), feature_library=library)
        model.fit(x_train, t=np.array(time_rec_obs), x_dot=dx_dot_train)
        
        ### SINDy Score Fitness Evaluation ###
        fitness = 1-model.score(x_train, t = np.array(time_rec_obs), x_dot = dx_dot_train)
        
        ### Simpsons Rule Fitness Evaluation
        # def dx_dt_total(t, x):
            # return (1/ mag_factor) * model.predict(np.array([[x[0], x[1], x[2], x[3], x[4]]]))[0] \
                   # + np.hstack([dxdt_expected(x[4], [x[0], x[1], x[2], x[3]]), 1.0])
        
        # sol = solve_ivp(dx_dt_total, y0=np.hstack([x0_nominal, t_start]), t_span=[t_true[0], time_rec_obs[-1]],
                        # t_eval=time_rec_obs, rtol=reltol, atol=abstol)
        # x_4fit = sol.y
        
        # if sol.status != 0:
            # print("Diverging behaviour, returning fitness of 1e99")
            # fitness = 1e99
        # else:
            # fitness = simps(np.square((x_4fit[0, :]-x_train[:,0])/np.max(x_train[:,0])) + np.square((x_4fit[1, :]
            # - x_train[:,1])/np.max(x_train[:,1])) + np.square((x_4fit[2, :] - x_train[:,2])/np.max(x_train[:,2]))
            # + np.square((x_4fit[3, :] - x_train[:,3])/np.max(x_train[:,3])), x=time_rec_obs)
        
        return fitness
    
    if flag_solution == 0:
        OptResults = optimize.minimize(coeff_estimation, omega, tol=1e-2, options={"maxiter": 10})
    else:
        OptResults = optimize.minimize(coeff_estimation, omega, tol=1e-2, options={"maxiter": 10})

    k_array = OptResults.x
    fitness = OptResults.fun

    if flag_solution == 1:
        print('Constants:')
        print(k_array)
        np.savetxt('output/k_array.txt', k_array, delimiter=',')

        print('Fitness:')
        print(fitness)
        
        def make_SINDy_func(tree, dimension):
            sr_func = lambda r, theta, vr, vt, t: toolbox.compile(tree)([r, theta], [vr, vt], t, *k_array)[dimension]
            return sr_func
        
        funcs = []
        for i in range(ntrees):
            for j in range(dimensions):
                funcs.append(make_SINDy_func(individual[i], j))
            
        library = ps.CustomLibrary(library_functions=funcs)
        model = ps.SINDy(feature_library=library, feature_names=var_list, optimizer=ps.STLSQ(threshold=1e-7))
        model.fit(x_train, t=np.array(time_rec_obs), x_dot=dx_dot_train)
        
        print("\n Divide Xi by e+0" + str(int(np.log10(mag_factor))) + "\n")
        model.print(precision=20)

        clims = [-30, 30]
        plt.figure()
        plt.imshow(sigma_plot_sparsity(model.coefficients().T), interpolation='nearest', cmap='GnBu')
        plt.clim(clims)
        plt.savefig('figures/coefficients_naAlpha.png', dpi=600)

        def dx_dt_total(t, x):
            return (1/ mag_factor) * model.predict(np.array([[x[0], x[1], x[2], x[3], x[4]]]))[0] + np.hstack([dxdt_expected(x[4], [x[0], x[1], x[2], x[3]]), 1.0])

        # Evolve the new initial condition in time with the SINDy model
        x_test_sim = solve_ivp(dx_dt_total, y0=np.hstack([x0_nominal, t_start]), t_span=[t_true[0], t_true[-1]], t_eval=t_true,
                           rtol=reltol,
                           atol=abstol).y
        
        plt.style.use("classic")
        fig, axs = plt.subplots(x_true.shape[0], 1, sharex=True, figsize=(10, 9))
        for i in range(x_true.shape[0]):
            axs[i].plot(t_true, x_test_sim[i, :], label='Estimated Model', linewidth=2)  # 'r--',
            axs[i].plot(t_true, x_true[i, :], color='#888888', linestyle = '--', label='True Model', linewidth=2)  # 'k'

            axs[i].scatter(0, posvelacc0_measured[i], color='green', label='Observations')
            ylabel = var_list[i]
            for j in range(len(time_rec_obs)):
                axs[i].scatter(time_rec_obs[j], x_train[j][i], color='green')
                
                axs[i].legend(loc = 0)
                xlabel = "time (s)"
                axs[i].set_xlabel(xlabel, fontsize=16)
                axs[i].set_ylabel(ylabel, fontsize=16)
                axs[i].grid(True)
                plt.tight_layout()
        plt.savefig('figures/solution_x.png')
        
        delta_x = x_true-x_test_sim[0:x_true.shape[0]]
        
        fig = plt.figure(figsize=(20,8), dpi= 75, facecolor='w', edgecolor='k')
        t_plot = t_true[0:round(len(t_true)*1)]
        x_plot = delta_x[0,0:round(len(t_true)*1)].transpose()
        plt.plot(t_plot, x_plot)
        plt.savefig('figures/solution_delta_x.png')

    return [fitness, ]


toolbox.register("evaluate", evalSymbReg)


toolbox.register("map", futures.map)

history = tools.History()
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=7))


def main():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=60)
    hof = tools.HallOfFame(1)
    # EVOLVE HERE:
    pop, log = my_eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.8, ngen=10, stats=mstats, halloffame=hof, verbose=True)

    print('Best individual : ')
    for i in range(ntrees):
        print(hof[0][i])
    evalSymbReg(hof[0], flag_solution=1)
    return pop, log, hof


if __name__ == '__main__':
    main()
