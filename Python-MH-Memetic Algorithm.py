############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Memetic Algorithm

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Memetic_Algorithm, File: Python-MH-Memetic Algorithm.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Memetic_Algorithm>

############################################################################

# Required Libraries
import pandas as pd
import numpy  as np
import math
import random
import os

# Function: Initialize Variables
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5]):
    population = pd.DataFrame(np.zeros((population_size, len(min_values))))
    population['Fitness'] = 0.0
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population.iloc[i,j] = random.uniform(min_values[j], max_values[j])
        population.iloc[i,-1] = target_function(population.iloc[i,0:population.shape[1]-1])
    return population

# Function: Fitness
def fitness_function(population): 
    fitness = pd.DataFrame(np.zeros((population.shape[0], 1)))
    fitness['Probability'] = 0.0
    for i in range(0, fitness.shape[0]):
        fitness.iloc[i,0] = 1/(1+ population.iloc[i,-1] + abs(population.iloc[:,-1].min()))
    fit_sum = fitness.iloc[:,0].sum()
    fitness.iloc[0,1] = fitness.iloc[0,0]
    for i in range(1, fitness.shape[0]):
        fitness.iloc[i,1] = (fitness.iloc[i,0] + fitness.iloc[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness.iloc[i,1] = fitness.iloc[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix = 0
    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness.iloc[i, 1]):
          ix = i
          break
    return ix

# Function: Offspring
def breeding(population, fitness, min_values = [-5,-5], max_values = [5,5], mu = 1, elite = 0):
    offspring = population.copy(deep = True)
    b_offspring = 0
    if (elite > 0):
        preserve = population.nsmallest(elite, "Fitness").copy(deep = True)
        for i in range(0, elite):
            for j in range(0, offspring.shape[1]):
                offspring.iloc[i,j] = preserve.iloc[i,j]
    for i in range (elite, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - 1):
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring.iloc[i,j] = np.clip(((1 + b_offspring)*population.iloc[parent_1, j] + (1 - b_offspring)*population.iloc[parent_2, j])/2, min_values[j], max_values[j])           
            if(i < population.shape[0] - 1):   
                offspring.iloc[i+1,j] = np.clip(((1 - b_offspring)*population.iloc[parent_1, j] + (1 + b_offspring)*population.iloc[parent_2, j])/2, min_values[j], max_values[j]) 
        offspring.iloc[i,-1] = target_function(offspring.iloc[i,0:offspring.shape[1]-1]) 
    return offspring

#Function: Crossover Hill Clibing
def xhc(offspring, fitness, min_values = [-5,-5], max_values = [5,5], mu = 1):
    offspring_xhc = pd.DataFrame(np.zeros((2, len(min_values))))
    offspring_xhc['Fitness'] = 0.0
    b_offspring = 0
    for _ in range (0, offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(offspring) - 1), 1)[0]
        for j in range(0, offspring.shape[1] - 1):
            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
            if (rand <= 0.5):
                b_offspring = 2*(rand_b)
                b_offspring = b_offspring**(1/(mu + 1))
            elif (rand > 0.5):  
                b_offspring = 1/(2*(1 - rand_b))
                b_offspring = b_offspring**(1/(mu + 1))       
            offspring_xhc.iloc[0,j] = np.clip(((1 + b_offspring)*offspring.iloc[parent_1, j] + (1 - b_offspring)*offspring.iloc[parent_2, j])/2, min_values[j], max_values[j])           
            offspring_xhc.iloc[1,j] = np.clip(((1 - b_offspring)*offspring.iloc[parent_1, j] + (1 + b_offspring)*offspring.iloc[parent_2, j])/2, min_values[j], max_values[j])           
        offspring_xhc.iloc[0,-1] = target_function(offspring_xhc.iloc[0,0:offspring_xhc.shape[1]-1])
        offspring_xhc.iloc[1,-1] = target_function(offspring_xhc.iloc[1,0:offspring_xhc.shape[1]-1]) 
        if (offspring_xhc.iloc[1,-1] < offspring_xhc.iloc[0,-1]):
            for k in range(0, offspring.shape[1]): 
                offspring_xhc.iloc[0, k] = offspring_xhc.iloc[1,k]            
        if (offspring.iloc[parent_1, -1] < offspring.iloc[parent_2, -1]):
            if (offspring_xhc.iloc[0,-1] < offspring.iloc[parent_1, -1]):
                for k in range(0, offspring.shape[1]): 
                    offspring.iloc[parent_1, k] = offspring_xhc.iloc[0,k]
        elif(offspring.iloc[parent_2, -1] < offspring.iloc[parent_1, -1]):
            if (offspring_xhc.iloc[0,-1] < offspring.iloc[parent_2, -1]):
                for k in range(0, offspring.shape[1]): 
                    offspring.iloc[parent_2, k] = offspring_xhc.iloc[0,k]
    return offspring

# Function: Mutation
def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5]):
    d_mutation = 0            
    for i in range (0, offspring.shape[0]):
        for j in range(0, offspring.shape[1] - 1):
            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
            if (probability < mutation_rate):
                rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
                if (rand <= 0.5):
                    d_mutation = 2*(rand_d)
                    d_mutation = d_mutation**(1/(eta + 1)) - 1
                elif (rand > 0.5):  
                    d_mutation = 2*(1 - rand_d)
                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
                offspring.iloc[i,j] = np.clip((offspring.iloc[i,j] + d_mutation), min_values[j], max_values[j])
        offspring.iloc[i,-1] = target_function(offspring.iloc[i,0:offspring.shape[1]-1])                        
    return offspring

# MA Function
def memetic_algorithm(population_size = 5, mutation_rate = 0.1, elite = 0, min_values = [-5,-5], max_values = [5,5], eta = 1, mu = 1, generations = 50):    
    count = 0
    population = initial_population(population_size = population_size, min_values = min_values, max_values = max_values)
    fitness = fitness_function(population)    
    elite_ind = population.iloc[population['Fitness'].idxmin(),:].copy(deep = True)
    
    while (count <= generations):
        
        print("Generation = ", count, " f(x) = ", elite_ind [-1])
        
        offspring = breeding(population, fitness, min_values = min_values, max_values = max_values, mu = mu, elite = elite) 
        offspring = xhc(offspring, fitness, min_values = min_values, max_values = max_values, mu = mu)
        population = mutation(offspring, mutation_rate = mutation_rate, eta = eta, min_values = min_values, max_values = max_values)
        
        fitness = fitness_function(population)
        if(elite_ind [-1] > population.iloc[population['Fitness'].idxmin(),:][-1]):
            elite_ind  = population.iloc[population['Fitness'].idxmin(),:].copy(deep = True) 
        
        count = count + 1 
        
    print(elite_ind )    
    return elite_ind 

######################## Part 1 - Usage ####################################

# Function to be Minimized (Rosenbrocks Valley). Solution ->  f(x) = 0; xi = 1
def target_function(variables_values = [0,0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value

ma = memetic_algorithm(population_size = 100, mutation_rate = 0.05, elite = 1, eta = 1,  mu = 1, min_values = [-5,-5,-5,-5], max_values = [5,5,5,5], generations = 400)
