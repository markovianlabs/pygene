from __future__ import division

import numpy as np 
from sys import exit
import matplotlib.pyplot as plt 
import random,pdb
import operator
import time 

class GeneticAlgorithm(object):

#------------------------------------------------------

	def __init__(self, nParams, nPop, nGen, p_m):
		"""

		"""
		self.nParams = nParams
		self.nPop = nPop
		self.nGen = nGen
		self.p_m = p_m

		self.xdata = 0
		self.ydata = 0
		self.ydataerr = 0

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

	def selection_roulette(self, weights):
		'''
		performs weighted selection or roulette wheel selection on a list
		and returns the index selected from the list
		'''
        # sort the weights in ascending order
		sorted_indexed_weights = sorted(enumerate(weights), key=operator.itemgetter(1));
		indices, sorted_weights = zip(*sorted_indexed_weights);
        # calculate the cumulative probability
		tot_sum=sum(sorted_weights)
		prob = [x/tot_sum for x in sorted_weights]
		cum_prob=np.cumsum(prob)
        # select a random a number in the range [0,1]
		random_num=random.random()

		for index_value, cum_prob_value in zip(indices,cum_prob):
			if random_num < cum_prob_value:
				return index_value


#------------------------------------------------------

	def crossover(self, parent1, parent2):
		mid = int(len(parent1)/2)
		child1 = parent1[:]
		child2 = parent2[:]
		child1[mid:] = parent2[mid:]
		child2[mid:] = parent1[mid:]
		return [child1, child2]

#------------------------------------------------------

	def mutation(self, individual):
		x = np.random.random()
		# if x>self.p_m:
			# return individual
		# else:
		# ind = np.random.randint(0, len(individual)-1)
		individual *= np.random.normal(1, self.p_m, len(individual))
		return individual

#------------------------------------------------------

	def run(self):
		# initializing a random set of chromosomes 
		# or a full population
		current_gen = self.initialize()
		BestFitnessTillNow = 1e-100
		BestIndividualTillnow = np.random.random(self.nParams)
		BestGeneration = 0

		# Repeating the following for the maximum number of generations
		for i in range(self.nGen):
			# calculating the fitness measure for all individuals in the current generation
			allfitness = self.find_fitness(current_gen) + np.random.random(self.nPop)*1e-100
			best_ind = np.argmax(allfitness)
			best_fitness = allfitness[best_ind]
			if best_fitness > BestFitnessTillNow:
				BestFitnessTillNow = best_fitness
				BestIndividualTillnow[:] = current_gen[:,best_ind]
				BestGeneration = i+1
				print "Best fit till now: %1.5e, %1.5e"%tuple(BestIndividualTillnow), \
						"%1.5e"%BestFitnessTillNow, "Generation: %i"%BestGeneration
			# selecting the two parents favouring the better fitness
			# selection_roulette function only returns the index of the selected individual

			parent1 = current_gen[:,self.selection_roulette(allfitness)]
			parent2 = current_gen[:,self.selection_roulette(allfitness)]

			# Repeating the following for the half the number of populations
			for j in range(int(self.nPop/2)):

				# Doing crossover of two parents to give two individuals
				# assigning the two individuals to the beginning and the end of the population array
				[i1, i2] = self.crossover(parent1, parent2)
				current_gen[:,j] = i1
				current_gen[:,j] = i2

				# Performing mutation of the newly crossovered individuals
				current_gen[:,j] = self.mutation(current_gen[:,j])
				current_gen[:,self.nPop-j-1] = self.mutation(current_gen[:,self.nPop-j-1])
		return [BestIndividualTillnow, BestFitnessTillNow, BestGeneration]

#------------------------------------------------------

#==============================================================================

if __name__=="__main__":
	print "Hello world!!"
	# gaobj = GeneticAlgorithm()
	# gaobj.run()

   
