import unittest
import os.path
import sys
import matplotlib as mpl
# mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np
import multiprocessing as mp
import multiprocessing.pool as mp_pool
# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import covid_inference

number_of_cpus = mp.cpu_count()

class TestInference(unittest.TestCase):

    def xest_random_walk_on_normal(self):
        mean = 7
        variance = 2

        normal_test = covid_inference.normal(mean,variance)
        number_of_samples = 10000
        initial_position = np.array([1.0])
        step_size = 7
        # proposal_covariance = np.array([[0.5]])
        thinning_rate = 1

        output = covid_inference.random_walk(normal_test,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        np.testing.assert_allclose(np.mean(output),mean,0.01)
        np.testing.assert_allclose(np.var(output),variance,0.01)

        # test that we get the expected number of samples
        np.testing.assert_almost_equal(output.shape[0],number_of_samples)

    def test_random_walk_on_multivariate_normal(self):
        mean = np.array([6,2])
        covariance_matrix = np.array([[2,1],[1,2]])

        multivariate_normal_test = covid_inference.multivariate_normal(mean,covariance_matrix)
        number_of_samples = 10000
        initial_position = np.array([1.0,1.0])
        step_size = 3.5
        proposal_covariance = np.array([[1,0.5],[0.5,1]])
        thinning_rate = 10

        output = covid_inference.random_walk(multivariate_normal_test,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             proposal_covariance=proposal_covariance,
                                             thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        np.testing.assert_allclose(np.mean(output[:,0]),mean[0],0.01)
        np.testing.assert_allclose(np.var(output[:,0]),covariance_matrix[0,0],0.1)
        np.testing.assert_allclose(np.mean(output[:,1]),mean[1],0.01)
        np.testing.assert_allclose(np.var(output[:,1]),covariance_matrix[1,1],0.1)

        # test that we get the expected number of samples
        np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        np.testing.assert_almost_equal(output.shape[1],2)
