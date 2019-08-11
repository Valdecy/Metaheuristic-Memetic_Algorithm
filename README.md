# Metaheuristic-Memetic_Algorithm
Memetic Algorithm with Lamarckian Learning (xhc - Crossover Hill Climbing) to Minimize Functions with Continuous Variables. Real Values Encoding. The function returns: 1) An array containing the used value(s) for the target function and the output of the target function f(x). For example, if the function f(x1, x2) is used, then the array would be [x1, x2, f(x1, x2)].  


* population_size = The population size. The Default Value is 5.

* elite = The quantity of best indivduals to be preserved. The quantity should be low to avoid being traped in local otima. The Default Value is 0.

* mutation_rate = Chance to occur a mutation operation. The Default Value is 0.1

* eta = Value of the mutation operator. The Default Value is 1.

* min_values = The minimum value that the variable(s) from a list can have. The default value is -5.

* max_values = The maximum value that the variable(s) from a list can have. The default value is  5.

* generations = The total number of iterations. The Default Value is 50.

* std = If the population standard deviation is less than the std, then th population is reinitialized. The Default Value is 0.1.

* mu = Value of the breed operator. The Default Value is 1.

* target_function = Function to be minimized.
