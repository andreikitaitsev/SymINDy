'''A sketch of general symbolicSYNDy class WITH inner classes'''


# Functions
class evolver(object):
	
	def __init__(self, population, toolbox_local,  cxpb, mutpb, ngen, \
		stats=None, halloffame=None, verbose=__debug__):
 		self.population = population
		self.toolbox_local = toolbox_local
		self.cxpb = cxpb
		self.mutpb = mutpb
		self.ngen = ngen
		self.stats = stats
		self.halloffame = halloffame
		self.verbose = verbose		

	def my_varAnd(population, toolbox_local, cxpb, mutpb):
	    '''
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
	    '''
	    # Create an offspring list sampled from the population   
	    offspring = [toolbox_local.clone(ind) for ind in population]

	    # Apply crossover and mutation on the offspring
	    for i in range(1, len(offspring), 2):
	        # for h_component in range(ntrees):
	        if random.random() < cxpb:
	            h_component = random.randint(0, ntrees-1) #where do we define ntrees?
	            offspring[i - 1][h_component], offspring[i][h_component] = toolbox_local.mate(\
	                offspring[i - 1][h_component], offspring[i][h_component])
	            del offspring[i - 1].fitness.values, offspring[i].fitness.values

	    for i in range(len(offspring)):
	        for h_component in range(ntrees):
	            if random.random() < mutpb:
	                # h_component = random.randint(0, ntrees-1)
	                offspring[i][h_component], = toolbox_local.mutate(offspring[i][h_component])
	                del offspring[i].fitness.values
	    return offspring

	def my_eaSimple(population, toolbox_local, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
	    '''
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
	    '''
	    logbook = tools.Logbook()
	    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

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

	def __call__(data):
		population, logbook = self.my_eaSimple(self.population, self.toolbox_local, self.cxpb,\
			self.mutpb, self.ngen, self.stats, self.halloffame, self.verbose)
		return population, logbook


class evaluator(object):
	def __init__(individual, dxdt):
		self.individual=individual
		self.dxdt=dxdt
		
	def evalSymbReg(individual):
	    '''Fitness function to evaluate symbolic regression.'''
	    # Transform the tree expression in a callable function
	    sr_functions = []
	    for i in range(ntrees):
	        sr_functions.append(toolbox.compile(expr=individual[i]))

	    library = ps.CustomLibrary(library_functions=sr_functions)
	    model = ps.SINDy(feature_library=library)
	    model.fit(x_train, t=np.array(time_rec_obs), x_dot=x_dot_train)
	   
	    #! Use R of thresholded least square
	    fitness = - model.score(x_train, t=np.array(time_rec_obs), x_dot=x_dot_train)
	    final_library_functions = []
	    for i in range(ntrees):
	        fn = lambda x: sr_functions[i](x, *k_array)  # x[0], x[1], x[2]
	        final_library_functions.append(fn)
	    custom_library = ps.CustomLibrary(library_functions=final_library_functions)
	    library = poly_library + custom_library

	    model = ps.SINDy(feature_library=library)
	    model.fit(x_train, t=np.array(time_rec_obs), x_dot=x_dot_train)

	    model.print()

	    x0_test = x0_nominal #[1.3, 5.12]
	    x_test = solve_ivp(dxdt, y0=x0_test, t_span=[t_true[0], t_true[-1]], t_eval=t_true, rtol=reltol,
	                       atol=abstol).y

	    #? why do we append t_start to x0_test?
	    # See comment above about the inclusion of time in the state.
	    x0_test.append(t_start)
	    x0_test = np.asarray(x0_test)

	    # Evolve the new initial condition in time with the SINDy model
	    x_test_sim = model.simulate(x0_test, t_true)
	    return [fitness, ]

    def __call__(data):
    	fitness=evalSymbReg(self.individual)
    	return fitness


class symbregrSINDy(object):
	def __init__(self, evolver, evaluator, parameters):
		'''
		Inputs:
			evolver - class instance, an algorithm, which defines how the population is being 
				evolved. Shall be callable.
			evaluator - class instance, called to evaluate the performance of pySINDy
				with the given set of library functions
		'''
		self.evolver = evolver
		self.evaluator = evaluator
	def fit(self):
		self.population, self.logbook = self.evlolver() 
		self.fitness = self.vvaluator(self.population, self.logbook)
	def main():
	    x_xdot_xddot = [position_rec_obs, velocity_rec_obs, acc_rec_obs]

	    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	    stats_size = tools.Statistics(len)
	    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	    mstats.register("avg", np.mean)
	    mstats.register("std", np.std)
	    mstats.register("min", np.min)
	    mstats.register("max", np.max)

	    pop = toolbox.population(n=300)  
	    hof = tools.HallOfFame(1)
	    # EVOLVE HERE:
	    pop, log = my_eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.8, ngen=5, stats=mstats, halloffame=hof, verbose=True)

	    expr = hof[0]
	    evalSymbReg(hof[0], flag_solution=1)

	    return pop, log, hof
