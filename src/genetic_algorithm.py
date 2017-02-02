import numpy as np 
from sys import exit
import matplotlib.pyplot as plt 


class GeneticAlgorithm:

#------------------------------------------------------

	def __init__(self, nParams=2, nPop=100, nGen=100):
		"""

		"""
		self.nParams = nParams
		self.nPop = nPop
		self.nGen = nGen

#------------------------------------------------------

	def initialize(self):
		return np.random.random((self.nParams, self.nPop))

#------------------------------------------------------

	def find_fitness(self, arr):
		allfitness = np.zeros((self.nPop))
		for i in range(self.nPop):
			allfitness[i] = self.fitness_function(arr[:,i])
		return allfitness

#------------------------------------------------------

	def fitness_function(self, params):
		return np.random.random()

#------------------------------------------------------

	def selection_roulette(self, allfitness):
		return 1

#------------------------------------------------------

	def crossover(self, parent1, parent2):
		return [np.zeros((self.nParams)), np.zeros((self.nParams))]

#------------------------------------------------------

	def mutation(self, individual):
		return np.zeros((self.nParams))

#------------------------------------------------------

	def run(self):
		# initializing a random set of chromosomes 
		# or a full population
		current_gen = self.initialize()

		# Repeating the following for the maximum number of generations
		for i in range(self.nGen):
			# calculating the fitness measure for all individuals in the current generation
			allfitness = self.find_fitness(current_gen)

			# selecting the two parents favouring the better fitness
			# selection_roulette function only returns the index of the selected individual
			parent1 = current_gen[:,self.selection_roulette(allfitness)]
			parent2 = current_gen[:,self.selection_roulette(allfitness)]

			# Repeating the following for the half the number of populations
			for j in range(self.nPop/2):

				# Doing crossover of two parents to give two individuals
				# assigning the two individuals to the beginning and the end of the population array
				[current_gen[:,j], current_gen[:,self.nPop-j-1]] = self.crossover(parent1, parent2)

				# Performing mutation of the newly crossovered individuals
				current_gen[:,j] = self.mutation(current_gen[:,j])
				current_gen[:,self.nPop-j-1] = self.mutation(current_gen[:,self.nPop-j-1])

		return 0

#------------------------------------------------------

#==============================================================================

gaobj = GeneticAlgorithm()
gaobj.run()

