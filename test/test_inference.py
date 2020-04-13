import unittest
import os.path
import sys
import pandas as pd
import seaborn as sns; sns.set(); sns.set_style("ticks", {'axes.spines.right': False,
                                                          'axes.spines.top': False})
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np
from scipy import stats
import multiprocessing as mp
import multiprocessing.pool as mp_pool
# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))

import covid_inference; import covid_models

number_of_cpus = mp.cpu_count()

class TestInference(unittest.TestCase):

    def xest_random_walk_on_normal(self):
        print("\nTesting random walk on a normal distribution")
        mean = 7
        variance = 2

        normal_test = covid_models.normal(mean,variance)
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

        plt.clf()
        x = np.linspace(mean - 3*np.sqrt(variance), mean + 3*np.sqrt(variance), 100)
        h = plt.plot(x, stats.norm.pdf(x, mean, np.sqrt(variance)),zorder=1)
        myhist = plt.hist(output,100,density=True)
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','1d_normal_random_walk_histogram.pdf'))

    def xest_random_walk_on_multivariate_normal(self):
        print("\nTesting random walk on a multivariate normal distribution")
        mean = np.array([6,2,18])
        covariance_matrix = np.eye(3)

        multivariate_normal_test = covid_models.multivariate_normal(mean,covariance_matrix)
        number_of_samples = 100000
        initial_position = np.array([1.0,1.0,1.0])
        step_size = 1.5
        # proposal_covariance = np.array([[1,0.5],[0.5,1]])
        thinning_rate = 1

        output = covid_inference.random_walk(multivariate_normal_test,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        np.testing.assert_allclose(np.mean(output[:,0]),mean[0],0.1)
        np.testing.assert_allclose(np.var(output[:,0]),covariance_matrix[0,0],0.1)
        np.testing.assert_allclose(np.mean(output[:,1]),mean[1],0.1)
        np.testing.assert_allclose(np.var(output[:,1]),covariance_matrix[1,1],0.1)

        # test that we get the expected number of samples
        np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        np.testing.assert_almost_equal(output.shape[1],len(initial_position))

        # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output[:-1:100],columns=['$x_1$','$x_2$','$x_3$']))
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_random_walk_multivariate_normal.pdf'))

    def xest_random_walk_on_hybrid_rosenbrock23(self):
        print("\nTesting random walk on a hybrid Rosenbrock (2,3) distribution")
        mean = 1
        shape = np.array([.05,5,5,5,5])

        rosenbrock_test = covid_models.hybrid_rosenbrock23(mean,shape)
        number_of_samples = 1000000
        initial_position = np.array([1.0,1.0,1.0,1.0,1.0])
        step_size = 0.2
        # proposal_covariance = np.array([[1,0.5],[0.5,1]])
        thinning_rate = 1

        output = covid_inference.random_walk(rosenbrock_test,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        # np.testing.assert_allclose(np.mean(output[:,0]),mean,0.1)

        # test that we get the expected number of samples
        np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        np.testing.assert_almost_equal(output.shape[1],len(initial_position))

        # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output[:-1:100],columns=['$x_1$','$x_2$','$x_3$','$x_4$','$x_5$']))
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_random_walk_rosenbrock.pdf'))

    def xest_mala_on_normal(self):
        print("\nTesting MALA on a normal distribution")
        mean = 7
        variance = 2

        normal_test = covid_models.normal(mean,variance)
        number_of_samples = 100000
        initial_position = np.array([1.0])
        step_size = 4.0
        # proposal_covariance = np.array([[0.5]])
        thinning_rate = 1

        output = covid_inference.mala(normal_test,
                                      number_of_samples,
                                      initial_position,
                                      step_size,
                                      thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        print('mean:',np.mean(output))
        print('variance:',np.var(output))
        np.testing.assert_allclose(np.mean(output),mean,0.01)
        np.testing.assert_allclose(np.var(output),variance,0.01)

        # test that we get the expected number of samples
        np.testing.assert_almost_equal(output.shape[0],number_of_samples)

        plt.clf()
        x = np.linspace(mean - 3*np.sqrt(variance), mean + 3*np.sqrt(variance), 100)
        h = plt.plot(x, stats.norm.pdf(x, mean, np.sqrt(variance)),zorder=1)
        myhist = plt.hist(output,100,density=True)
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','1d_normal_mala_histogram.pdf'))

    def xest_mala_on_multivariate_normal(self):
        print("\nTesting MALA on a multivariate normal distribution")
        mean = np.array([6,2,18])
        covariance_matrix = np.eye(3)

        multivariate_normal_test = covid_models.multivariate_normal(mean,covariance_matrix)
        number_of_samples = 100000
        initial_position = np.array([1.0,1.0,1.0])
        step_size = 1.5
        # proposal_covariance = np.array([[1,0.5],[0.5,1]])
        thinning_rate = 1

        output = covid_inference.mala(multivariate_normal_test,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        np.testing.assert_allclose(np.mean(output[:,0]),mean[0],0.1)
        np.testing.assert_allclose(np.var(output[:,0]),covariance_matrix[0,0],0.1)
        np.testing.assert_allclose(np.mean(output[:,1]),mean[1],0.1)
        np.testing.assert_allclose(np.var(output[:,1]),covariance_matrix[1,1],0.1)

        # test that we get the expected number of samples
        np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        np.testing.assert_almost_equal(output.shape[1],len(initial_position))

        # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output[:-1:100],columns=['$x_1$','$x_2$','$x_3$']))
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_mala_multivariate_normal.pdf'))

    def xest_mala_on_hybrid_rosenbrock23(self):
        print("\nTesting MALA on a hybrid Rosenbrock (2,3) distribution")
        mean = 1
        shape = np.array([.05,5,5,5,5])

        rosenbrock_test = covid_models.hybrid_rosenbrock23(mean,shape)
        number_of_samples = 1000000
        initial_position = np.array([1.0,1.0,1.0,1.0,1.0])
        step_size = 0.07
        # proposal_covariance = np.array([[1,0.5],[0.5,1]])
        thinning_rate = 1

        output = covid_inference.mala(rosenbrock_test,
                                      number_of_samples,
                                      initial_position,
                                      step_size,
                                      thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        # np.testing.assert_allclose(np.mean(output[:,0]),mean,0.1)

        # test that we get the expected number of samples
        np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        np.testing.assert_almost_equal(output.shape[1],len(initial_position))

        # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output[:-1:10],columns=['$x_1$','$x_2$','$x_3$','$x_4$','$x_5$']))
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_mala_rosenbrock.pdf'))

    def xest_simple_manifold_mala_on_multivariate_normal(self):
        print("\nTesting simple manifold MALA on a multivariate normal distribution")
        mean = np.array([6,2,18])
        covariance_matrix = np.eye(3)

        multivariate_normal_test = covid_models.multivariate_normal(mean,covariance_matrix)
        number_of_samples = 1000
        initial_position = np.array([5.0,1.0,13.0])
        step_size = 3.5
        # proposal_covariance = np.array([[1,0.5],[0.5,1]])
        thinning_rate = 10

        output = covid_inference.simple_manifold_mala(multivariate_normal_test,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             regularization_constant=1,
                                             thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        print('mean 1',np.mean(output[:,0]))
        print('mean 2',np.mean(output[:,1]))
        print('mean 3',np.mean(output[:,2]))
        np.testing.assert_allclose(np.mean(output[:,0]),mean[0],0.1)
        np.testing.assert_allclose(np.var(output[:,0]),covariance_matrix[0,0],0.1)
        np.testing.assert_allclose(np.mean(output[:,1]),mean[1],0.1)
        np.testing.assert_allclose(np.var(output[:,1]),covariance_matrix[1,1],0.1)
        np.testing.assert_allclose(np.mean(output[:,2]),mean[2],0.1)
        np.testing.assert_allclose(np.var(output[:,2]),covariance_matrix[2,2],0.1)

        # test that we get the expected number of samples
        np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        np.testing.assert_almost_equal(output.shape[1],len(initial_position))

        # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output,columns=['$x_1$','$x_2$','$x_3$']))
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_sm_mala_multivariate_normal.pdf'))

        plt.clf()
        for i in range(3):
            x = np.linspace(mean[i] - 3*np.sqrt(covariance_matrix[i,i]), mean[i] + 3*np.sqrt(covariance_matrix[i,i]), 100)
            h = plt.plot(x, stats.norm.pdf(x, mean[i], np.sqrt(covariance_matrix[i,i])),zorder=i)
        plt.hist(output,100,density=True)
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','multivariate_normal_random_walk_histogram.pdf'))

    def xest_simple_manifold_mala_on_rosenbrock(self):
        print("\nTesting smMALA on a hybrid Rosenbrock (2,3) distribution")
        mean = 1
        shape = np.array([.05,5,5,5,5])

        rosenbrock_test = covid_models.hybrid_rosenbrock23(mean,shape)
        number_of_samples = 100000
        initial_position = np.array([1.0,1.0,1.0,1.0,1.0])
        step_size = 0.32
        thinning_rate = 1

        output = covid_inference.simple_manifold_mala(rosenbrock_test,
                                                      number_of_samples,
                                                      initial_position,
                                                      step_size,
                                                      thinning_rate=thinning_rate,)

        # test that mean and variance inferences are within 0.01 of the ground truth
        print(np.mean(output[:,0]))
        # np.testing.assert_allclose(np.mean(output[:,0]),mean,0.1)

        # test that we get the expected number of samples
        np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        np.testing.assert_almost_equal(output.shape[1],len(initial_position))

        # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output[:-1:10],columns=['$x_1$','$x_2$','$x_3$','$x_4$','$x_5$']))
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_sm_mala_rosenbrock.pdf'))

        plt.clf()
        x = np.linspace(mean - 3*np.sqrt(0.1), mean + 3*np.sqrt(0.1), 100)
        h = plt.plot(x, stats.norm.pdf(x, mean, np.sqrt(0.1)))
        plt.hist(output[:,0],100,density=True)
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','rosenbrock_sm_mala_histogram.pdf'))

    def xest_random_walk_on_poisson(self):
        print("\nTesting random walk on a poisson distribution with data")

        # make some data
        n = 100
        beta = np.array([1,2])
        days = np.zeros((n,len(beta)))
        days[:,0] = 1
        days[:,1] = np.arange(0,1,.01)
        mu = np.exp(days.dot(beta))
        data = stats.poisson.rvs(mu)

        poisson_test = covid_models.poisson_data(data,days)
        number_of_samples = 100000
        initial_position = np.array([0.92784717, 2.13682143])
        step_size = 0.1
        # proposal_covariance = np.diag([20.0,0.02,20.0])
        # proposal_covariance = np.array([[1,0.5],[0.5,1]])
        thinning_rate = 1

        output = covid_inference.random_walk(poisson_test,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        # np.testing.assert_allclose(np.mean(output[:,0]),mean[0],0.1)
        # np.testing.assert_allclose(np.var(output[:,0]),covariance_matrix[0,0],0.1)

        # test that we get the expected number of samples
        # np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        # np.testing.assert_almost_equal(output.shape[1],len(initial_position))
        print('x_1',np.mean(output[:,0]))
        print('x_2',np.mean(output[:,1]))
        # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output,columns=['$x_1$','$x_2$']))
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_random_walk_poisson.pdf'))

    def test_random_walk_on_negative_binomial(self):
        print("\nTesting random walk on a negative binomial distribution with data")

        # make some data
        n = 24
        beta = np.array([1,2])
        days = np.zeros((n,len(beta)))
        days[:,0] = 1
        days[:,1] = np.linspace(0,1,24)
        mu = np.exp(days.dot(beta))
        data = stats.poisson.rvs(mu)

        negative_binomial_test = covid_models.negative_binomial_data(data)
        number_of_samples = 10000
        initial_position = np.array([1, 2, 2])
        step_size = 0.01
        # proposal_covariance = np.diag([20.0,0.02,20.0])
        # proposal_covariance = np.array([[1,0.5],[0.5,1]])
        thinning_rate = 1

        output = covid_inference.random_walk(negative_binomial_test,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        # np.testing.assert_allclose(np.mean(output[:,0]),mean[0],0.1)
        # np.testing.assert_allclose(np.var(output[:,0]),covariance_matrix[0,0],0.1)

        # test that we get the expected number of samples
        # np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        # np.testing.assert_almost_equal(output.shape[1],len(initial_position))
        print('x_1',np.mean(output[:,0]))
        print('x_2',np.mean(output[:,1]))
        print('x_3',np.mean(output[:,2]))
        # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output,columns=['$x_1$','$x_2$','$x_3$']))
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_random_walk_negative_binomial.pdf'))
