import pycxsimulator
from pylab import *

# Constants and parameters
population_size = 20
vmin = -5
vmax = 5
search_space_size = vmax - vmin
max_initial_velocity = 0.1 * search_space_size
min_initial_velocity = 0.01 * search_space_size
problem_dimension = 2
inertia_weight = 0.4
c1 = 0.3
c2 = 0.3
plot_linspace = linspace(vmin,vmax,1001)

def fitness_function(x):
    return sum(x**2 - 10*cos(2*pi*x) + 10, axis=-1)

xvalues, yvalues = meshgrid(plot_linspace, plot_linspace)
zvalues = fitness_function(dstack([xvalues, yvalues]))

def initialize():
    global population, population_velocity, population_fitness, time, pbest, pbest_fitness, gbest_index

    time = 0
    population = random(size=[population_size, problem_dimension])*search_space_size + vmin
    population_velocity = random(size=[population_size, problem_dimension]) * \
                              (max_initial_velocity-min_initial_velocity) + \
                              min_initial_velocity
    velocity_direction_mask = random(size=[population_size, problem_dimension]) > 0.5
    population_velocity[velocity_direction_mask] = - population_velocity[velocity_direction_mask]
    population_fitness = fitness_function(population)
    pbest = array(population)
    pbest_fitness = array(population_fitness)
    gbest_index = argmin(pbest_fitness)


def observe():
    global population, population_velocity, population_fitness, time, pbest, pbest_fitness, gbest_index
    cla()
    contourf(xvalues, yvalues, zvalues)
    if len(gcf().axes) == 1:
        colorbar()
    for i in range(population_size):
        quiver(population[i,0], population[i,1], population_velocity[i,0], population_velocity[i,1], scale=search_space_size)
    scatter(population[:,0], population[:,1], c="#FF0000")
    scatter(0, 0, c="#FFFFFF", marker="x")

    xlabel(r'$x_1$', fontsize=16)
    ylabel(r'$x_2$', fontsize=16)
    title('Find '+r'$min(f(x_1, x_2))$'+'. t = ' + str(time) + ', gbest_fitness = ' + "{:.2f}".format(pbest_fitness[gbest_index]), fontsize=24)

def update():
    global population, population_velocity, population_fitness, time, pbest, pbest_fitness, gbest_index

    time +=1
    population_velocity = inertia_weight*population_velocity + \
                          c1*random(size=population.shape)*(pbest-population) + \
                          c2*random(size=population.shape)*(pbest[gbest_index]-population)
    population += population_velocity
    population_fitness = fitness_function(population)
    pbest_mask = population_fitness < pbest_fitness
    pbest[pbest_mask] = population[pbest_mask]
    pbest_fitness[pbest_mask] = population_fitness[pbest_mask]
    gbest_index = argmin(pbest_fitness)

pycxsimulator.GUI().start(func=[initialize, observe, update])
