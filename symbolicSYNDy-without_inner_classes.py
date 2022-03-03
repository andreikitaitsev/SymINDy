import operator
import numpy as np
import random
import pysindy as ps

from scoop import futures
from deap import base, creator, gp, tools


from scipy.integrate import solve_ivp
from scipy.integrate import simps
from scipy import optimize

from python_utils import *

from sklearn.metrics import *
import matplotlib.pyplot as plt
import matplotlib as mpl

import os

# the library class will be here
class library():
    def __init__(self, polynomial=None, trigonometric=None,\
        fourier=None, nc=1, dimensions=1):


class symbregrSINDy(object):
    def __init__(
        self,
        ngen=5,
        ntrees=5,
        mutpb=0.8,
        cxpb=0.7,
        dims=1,
        library=None,
        seed=0,
        score_metrics=None,
        score_metrics_kwargs=None,
        sindy_kwargs=None,
    ):
        """
        Inputs:
                score_metrics - the metrics to use in pySINDy model.score. Default - None, uses
                        R2 coefficient of determination (see sindy model score https://pysindy.readthedocs.io/en/latest/api/pysindy.html
                        and sklearn model evaluation for reference, https://scikit-learn.org/stable/modules/model_evaluation).
                score_metrics_kwargs - key value arguments for scoring function. If None, uses default pySINDy score kwargs.
        """
        self.ngen = ngen
        self.ntrees = ntrees
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.dims = dims
        self.library = library  # create library object here
        self.seed = seed
        random.seed(seed)
        # add verbal encoding of different scoring functions
        self.score_metrics = score_metrics
        self.score_metrics_kwargs = score_metrics_kwargs

    @staticmethod   
    def configure_DEAP(ntrees=5, nc=1, dimensions=1):
        """
        Inputs:
                ntrees -int, number of trees defining an individual. Defualt=5.
                nc -int, number of symbolic constants associated to the individual. Defualt=1.
                dimensions - int, Defualt = 1.
        """

        def _random_mating_operator(ind1, ind2):
            roll = random.random()
            if roll < 0.5:
                return gp.cxOnePoint(ind1, ind2)
            elif roll < 1.5:
                return gp.cxOnePointLeafBiased(ind1, ind2, termpb=0.5)

        def _random_mutation_operator(individual):
            roll = random.random()
            if roll < 0.5:
                return gp.mutInsert(individual, pset=pset)
            elif roll < 0.66:
                return gp.mutShrink(individual)
            elif roll < 2.66:
                return gp.mutNodeReplacement(individual, pset=pset)

        size_input = 1 + nc
        intypes = [float for i in range(size_input)]
        # Create a primitive set
        pset = gp.PrimitiveSetTyped(
            "MAIN", intypes, float
        )  # 1)name, 2)type of each input, 3)type of the output
        pset.addPrimitive(np.multiply, [float, float], float, name="mul")
        pset.addPrimitive(np.sin, [float], float, name="sin")
        pset.addPrimitive(np.cos, [float], float, name="cos")
        pset.addPrimitive(np.square, [float], float, name="square")
        pset.renameArguments(ARG0="y")
        pset.renameArguments(ARG1="p1")
        # add time as an independent variable, if necessary
        # pset.renameArguments(ARG2='t') # user defined
        creator.create(
            "FitnessMin", base.Fitness, weights=(-1.0,)
        )  # weights=-1 to indicate minimization
        creator.create(
            "Subindividual", gp.PrimitiveTree
        )  # subindividual is a primitive tree which is populated from pset
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register(
            "expr", gp.genHalfAndHalf, pset=pset, type_=pset.ret, min_=1, max_=2
        )  # ? type_=pset.ret is the default of gp.genHalfAndHalf
        toolbox.register(
            "subindividual", tools.initIterate, creator.Subindividual, toolbox.expr
        )
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.subindividual,
            n=ntrees,
        )
        toolbox.register(
            "population", tools.initRepeat, list, toolbox.individual
        )  # ? the arguments to tools.initRepeat: container, func,
        # n; container=list, func=toolbox.individual n=? (what is n?)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate", _random_mating_operator)
        toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=1)  # ? never used
        toolbox.register("mutate", _random_mutation_operator)
        toolbox.register("map", futures.map)
        history = tools.History()
        toolbox.decorate("mate", history.decorator)
        toolbox.decorate("mutate", history.decorator)
        toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=2)
        )
        toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=2)
        )
        return toolbox, creator, pset, history

    def evalSymbReg(
        self, individual, x_train, x_dot_train, time_rec_obs=None, sindy_kwargs=None
    ):
        """Fitness function to evaluate symbolic regression.
        For additional documentation see SINDy model docs
        https://pysindy.readthedocs.io/en/latest/api/pysindy.html#module-pysindy.pysindy
        Inputs:
                individual -
                x_train - np array, training data
                x_dot_train - precomputed derivatives of the training data, optional. Defualt=None, no
                        precomputed derivatives (SINDY computes it using specified differentiation method).
                time_rec_obs - (float, numpy array of shape (n_samples,), or list of numpy arrays, optional (default None)) –
                        If t is a float, it specifies the timestep between each sample.
                        If array-like, it specifies the time at which each sample was collected.
                sindy_kwargs - dictionary with kwargs for SINDY. Default=None, no kwargs
        Outputs:
                [fitness] - list with fitness value. NB - DEAP requires output to be iterable (so, it shall be
                        a tuple or a list).
        """

        # Transform the tree expression in a callable function
        sr_functions = []
        for i in range(ntrees):
            sr_functions.append(toolbox.compile(expr=individual[i]))
        library = ps.CustomLibrary(library_functions=sr_functions)

        if sindy_kwargs is not None:
            model = ps.SINDy(feature_library=library, **sindy_kwargs)
        elif sindy_kwargs is None:
            model = ps.SINDy(feature_library=library, **sindy_kwargs)
        if x_dot_train is not None:
            model.fit(x_train, t=np.array(time_rec_obs), x_dot=x_dot_train)
        elif x_dot_train is None:
            model.fit(x_train, t=np.array(time_rec_obs))

        #! Uses corr coef of thresholded least square
        fitness = -model.score(
            x_train,
            t=np.array(time_rec_obs),
            x_dot=x_dot_train,
            t=time_rec_obs,
            u=None,
            multiple_trajectories=False,
            metric=self.score_metrics,
            **self.score_metrics_kwargs
        )

        # store trained model as an attribute #? Maybe change in the future for efficacy
        self.model = model
        return [
            fitness,
        ]

    @staticmethod
    def add_evalfunc_to_toolbex(
        toolbox, eval_func, x_train, x_dot_train, time_rec_obs, sindy_kwargs
    ):
        toolbox.register(
            "evaluate", eval_func, x_train, x_dot_train, time_rec_obs, sindy_kwargs
        )
        return toolbox

    # ? consider making it a staticmethod
    def my_eaSimple(
        self,
        population,
        toolbox_local,
        cxpb,
        mutpb,
        ngen,
        stats=None,
        halloffame=None,
        verbose=__debug__,
    ):
        """
        Takes in a population and evolves it in place using the varAnd() method.
        Returns the optimized population and a Logbook with the statistics of the evolution.

        Inputs:
                population – A list of individuals
                toolbox – A DEP Toolbox class instance, that contains the evolution operators.
                cxpb – The probability of mating two individuals.
                mutpb – The probability of mutating an individual.
                ngen – The number of generation.
                stats – A DEAP Statistics object that is updated inplace. Default=None.
                halloffame – A DEAP HallOfFame object that will contain the best individuals. Default=None.
                verbose – Whether or not to log the statistics. Default=__debug__.
        Outputs:
                population: The final population
                logbook - a logbook object with the statistics of the evolution.

        Pseudo code of eaSimple from DEAP
        evaluate(population)
                for g in range(ngen):
                        population = select(population, len(population))
                        offspring = varAnd(population, toolbox, cxpb, mutpb)
                        evaluate(offspring)
                        population = offspring
        """

        def _my_varAnd(population, toolbox_local, cxpb, mutpb):
            """
            Part of an evolutionary algorithm applying only the variation part (crossover and mutation).
            See https://deap.readthedocs.io/en/master/api/algo.html#deap.algorithms.varAnd for the reference.
            Inputs:
                    population - a list of individuals to vary. It is recommended that the population is created
                            with the toolbox.register method of toolbox object instance from DEAP
                    toolbox_local
                    cxpb - float, is the probability with which two individuals are crossed
                    mutpb - float, is the probability for mutating an individual
            Outputs:
                    offspring - a "list" of varied individuals that are independent of their parents (deepcopied)
            """
            # Create an offspring list sampled from the population
            offspring = [toolbox_local.clone(ind) for ind in population]

            # Apply crossover and mutation on the offspring
            for i in range(1, len(offspring), 2):
                # for h_component in range(ntrees):
                if random.random() < cxpb:
                    h_component = random.randint(
                        0, ntrees - 1
                    )  # where do we define ntrees?
                    (
                        offspring[i - 1][h_component],
                        offspring[i][h_component],
                    ) = toolbox_local.mate(
                        offspring[i - 1][h_component], offspring[i][h_component]
                    )
                    del offspring[i - 1].fitness.values, offspring[i].fitness.values

            for i in range(len(offspring)):
                for h_component in range(ntrees):
                    if random.random() < mutpb:
                        # h_component = random.randint(0, ntrees-1)
                        (offspring[i][h_component],) = toolbox_local.mutate(
                            offspring[i][h_component]
                        )
                        del offspring[i].fitness.values
            return offspring

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        # Evaluate the fitness of the first population
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
            offspring = _my_varAnd(offspring, toolbox_local, cxpb, mutpb)

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

    @staticmethod
    def init_stats():
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        return mstats

    def fit(self, x_train, x_dot_train=None, time_rec_obs=None):
        """Essentially, the key function, which calls everything else"""
        # Initiate DEAP
        toolbox, creator, pset, history = self.configure_DEAP(
            ntrees=self.ntrees, nc=self.nc, dimensions=self.dims
        )  # Add arguments from init
        toolbox = self.add_evalfunc_to_toolbex(
            toolbox, self.evalSymbReg, x_train, x_dot_train, time_rec_obs, self.sindy_kwargs
        )
        mstats = self.init_stats()
        pop = toolbox.population(n=300)
        hof = tools.HallOfFame(1)
        self.toolbox = toolbox
        self.creator = creator
        self.pset = pset
        self.history = history

        # Run the evolution
        pop, log = self.my_eaSimple(
            pop,
            self.toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.ngen,
            stats=mstats,
            halloffame=hof,
            verbose=self.verbose,
        )
        self.pop = pop
        self.log = log
        self.hof = hof

        # Store time_rec_obs as an attribute
        self.time_rec_obs = time_rec_obs

    def score(
        self,
        x,
        t=None,
        x_dot=None,
        u=None,
        multiple_trajectories=False,
        metric=r2_score,
        metric_kwargs=None,
    ):
        """
        Produce the score using trained model. The input data usually is a test set (is scoring does not
        exploit cross-validation).
        See pySINDy model score documentation for more details
        https://pysindy.readthedocs.io/en/latest/_modules/pysindy/pysindy.html#SINDy.score
        Inputs:
                x - data to evauate the model on (usually, test set).
                t - float, numpy array of shape (n_samples,), or list of numpy arrays, optional
                        (default None)
                x_dot: array-like or list of array-like, shape (n_samples, n_input_features),
                        optional (default None)
                u: array-like or list of array-like, shape(n_samples, n_control_features),
                        optional (default None)
                multiple_trajectories: boolean, optional (default False)
                        If True, x contains multiple trajectories and must be a list of
                        data from each trajectory. If False, x is a single trajectory.
                metric: callable, optional
                        Metric function with which to score the prediction. Default is the
                        R^2 coefficient of determination.
                metric_kws: dict, optional
                        Optional keyword arguments to pass to the metric function.
        Outputs:
                score - float, score metrics
        """
        if metric_kwargs is None:
            metric_kwargs = {}
        # Use R2
        fitness = - self.model.score(
            x,
            t = np.array(self.time_rec_obs),
            x_dot = x_dot,
            u = u,
            multiple_trajectories = multiple_trajectories,
            metric  =metric,
            **metric_kwargs
        )
        return fitness

    def predict(self, x, u=None, multiple_trajectories=False, x_dot_pred_kwargs = None, 
        x_pred_kwargs = None):
        """
		Predict Predict the time derivatives using the SINDy model.
		See pySINDy model.predict for more documentation.
		https://pysindy.readthedocs.io/en/latest/api/pysindy.html#pysindy.pysindy.SINDy.predict		
		Inpts:
			x: array-like or list of array-like, shape (n_samples, n_input_features)
				Samples
			u: array-like or list of array-like, shape(n_samples, n_control_features), \
					(default None)
				Control variables. If ``multiple_trajectories==True`` then u
				must be a list of control variable data from each trajectory. If the
				model was fit with control variables then u is not optional.
			multiple_trajectories: boolean, optional (default False)
				If True, x contains multiple trajectories and must be a list of
				data from each trajectory. If False, x is a single trajectory.
            x_pred_kwargs - dictionary of kwargs for pysindy.simulate function
            x_dot_pred_kwargs - dictonary fof kwargs for pysindy.predict function
		Outputs:
            x_pred – Simulation results using pysindy simulate
			x_dot_pred: array-like or list of array-like, shape (n_samples, n_input_features)
			 	Predicted time derivatives
		"""
        if x_pred_kwargs is not None:
            x_pred = self.model.silulate(x0, t, u=None, **x_pred_kwargs)
        else:
            x_pred = self.model.silulate(x0, t, u=None)
        if x_dot_pred_kwargs is not None:
            x_dot_pred = self.model.predict(x, u, multiple_trajectories, **x_dot_pred_kwargs)
        else:
            x_pred = self.model.silulate(x0, t, u=None) 
        return x_pred, x_dot_pred 

    def plot():
        pass
