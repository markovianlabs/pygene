"""
Example demonstrating the usage for the linear-fit scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../src')
from genetic_algorithm import GeneticAlgorithm

#----------------------------------------------------------

class LinearFit(GeneticAlgorithm):
	"""
	Extends the original MCMC class to sample the parameters of a linear model.

	Parameters
	-----------
	MCMC (Class): Parent MCMC class.
	m (Float): Feducial value of the slope of the linear data.
	c (Float): Feducial value of the intercept of the linear data.
	RedStd (Float): Feducial value of the standard deviation of the linear data.
	"""
	def __init__(self, m=25.0, c=25.0, RedStd=5.0, randomseed=250192):
		"""
		Instantiates the class by synthetically generating data.
		"""
		GeneticAlgorithm.__init__(self, nParams=2, nPop=100, nGen=1000, p_m=0.05)

		self.X=np.linspace(-10, 10, 25)
		self.delta = np.random.uniform(low=-1*RedStd, high=RedStd, size=len(self.X))
		self.Y = (m*self.X + c) + self.delta

#----------------------------------------------------------

	def initialize(self):
		return np.random.uniform(low=-50.0, high=50.0, size=(self.nParams, self.nPop))

#----------------------------------------------------------

	def FittingFunction(self, Params):
		"""
		Parametric form of the model.

		Parameters
		----------
		Params (1d array): Numpy array containing values of the parameters. 

		Returns
		-------
		model values (y = mx + c)
		"""
		return Params[0]*self.X + Params[1]

#----------------------------------------------------------

	def fitness_function(self, Params):
		"""
		Computes Chi-square.

		Parameters
		----------
		Params (1d array): Numpy array containing values of the parameters. 

		Returns
		-------
		chi square.
		"""
		kisquare = ((self.Y-self.FittingFunction(Params))/self.delta)**2
		return 1.0/np.sum(kisquare)*len(self.X)

#==============================================================================

if __name__=="__main__":
	co = LinearFit()
	print co.run()


