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
from scipy.integrate import solve_ivp
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

    def xest_simplified_manifold_mala_on_multivariate_normal(self):
        print("\nTesting simplified manifold MALA on a multivariate normal distribution")
        mean = np.array([6,2,18])
        covariance_matrix = np.eye(3)

        multivariate_normal_test = covid_models.multivariate_normal(mean,covariance_matrix)
        number_of_samples = 1000
        initial_position = np.array([5.0,1.0,13.0])
        step_size = 3.5
        # proposal_covariance = np.array([[1,0.5],[0.5,1]])
        thinning_rate = 10

        output = covid_inference.simplified_manifold_mala(multivariate_normal_test,
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

    def xest_simplified_manifold_mala_on_rosenbrock(self):
        print("\nTesting smMALA on a hybrid Rosenbrock (2,3) distribution")
        mean = 1
        shape = np.array([.05,5,5,5,5])

        rosenbrock_test = covid_models.hybrid_rosenbrock23(mean,shape)
        number_of_samples = 100000
        initial_position = np.array([1.0,1.0,1.0,1.0,1.0])
        step_size = 0.32
        thinning_rate = 1

        output = covid_inference.simplified_manifold_mala(rosenbrock_test,
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
        n = 25
        beta = np.array([1,0.4])
        days = np.zeros((n,len(beta)))
        days[:,0] = 1
        days[:,1] = np.arange(0,25,1)
        mu = np.exp(days.dot(beta))
        data = stats.poisson.rvs(mu)
        print(data)

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

    def xest_random_walk_on_negative_binomial(self):
        print("\nTesting random walk on a negative binomial distribution with data")

        # make some data
        n = 20
        beta = np.array([1,0.4,60])
        days = np.zeros((n,2))
        days[:,0] = 1
        days[:,1] = np.arange(0,len(days[:,0]))
        mu = np.exp(days.dot(beta[:2]))
        od = beta[2]*np.ones(len(days[:,1]))
        r=mu/(od-1)
        p = 1/od
        data = stats.nbinom.rvs(r,p)
        print('data:', data)

        negative_binomial_test = covid_models.negative_binomial_data(data)
        number_of_samples = 100000
        initial_position = np.array([ 1.13897153,  0.3940096 , 56.45188604])
        step_size = 0.02
        proposal_covariance = np.array([[ 1.99563291e-02, -1.18693670e-03,  5.86064693e-04],
                                        [-1.18693670e-03,  8.29099377e-05, -2.06102988e-06],
                                        [ 5.86064693e-04, -2.06102988e-06,  1.24682610e-01]]
)
        # proposal_covariance = np.array([[1,0.5],[0.5,1]])
        thinning_rate = 1

        output = covid_inference.random_walk(negative_binomial_test,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             thinning_rate=thinning_rate)
        print(np.cov(output[:10000].T))

        # test that mean and variance inferences are within 0.01 of the ground truth
        # np.testing.assert_allclose(np.mean(output[:,0]),mean[0],0.1)
        # np.testing.assert_allclose(np.var(output[:,0]),covariance_matrix[0,0],0.1)

        # test that we get the expected number of samples
        # np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        # np.testing.assert_almost_equal(output.shape[1],len(initial_position))
        print('x_1 mean:',np.mean(output[:,0]))
        print('x_2 mean:',np.mean(output[:,1]))
        print('x_3 mean:',np.mean(output[:,2]))

        plt.clf()
        fig, ax = plt.subplots(3,1,figsize=(10,10))
        for i in range(output.shape[1]):
            ax[i].plot(output[:,i])
            ax[i].set_xlabel('$\\beta_{}$'.format(i))
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','traceplots_random_walk_negative_binomial.pdf'))
        #
        # # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output[:-1:10],columns=['$x_1$','$x_2$','$x_3$']))
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_random_walk_negative_binomial.pdf'))

        thinning_rate = 100
        ys = np.zeros((number_of_samples//thinning_rate,len(days[:,0])))

        plt.clf()
        plt.figure(figsize=(8,6))
        for i in range(0,number_of_samples,thinning_rate):
            mu = np.exp(days.dot(output[i,:2]))
            od = output[i,2]*np.ones(len(days[:,0]))
            p = 1/od
            r = mu/(od-1)
            ys[i//thinning_rate] = np.random.negative_binomial(r,p)
            # plt.plot(days[:,1],ys[i//thinning_rate],linestyle='',marker='o',c='blue',alpha = 0.005,lw=3,zorder=1)
        mu = np.exp(days.dot(np.mean(output[:,:2],axis=0)))
        lower_error = np.percentile(ys,2.5,axis=0)
        upper_error = np.percentile(ys,97.5,axis=0)

        plt.plot(days[:,1],mu,linestyle='-',c='m')
        plt.plot(days[:,1],lower_error,linestyle='--',c='m')
        plt.plot(days[:,1],upper_error,linestyle='--',c='m')
        # plt.errorbar(days[:,1],np.mean(ys,axis=0),
        #              yerr=[lower_error ,upper_error],
        #              ecolor='r',fmt='none',capsize=2)
        plt.scatter(days[:,1],data,c='k',label='Data',edgecolors=None,s=10,zorder=2)
        # plt.xlim([0,12])
        # plt.ylim([0,250])
        plt.xlabel('Days')
        plt.ylabel('New cases')
        plt.title('Synthetic Data')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','time_series.pdf'))

    def xest_mala_on_negative_binomial(self):
        print("\nTesting MALA on a negative binomial distribution with data")

        # make some data
        n = 20
        beta = np.array([1,0.4,60])
        days = np.zeros((n,2))
        days[:,0] = 1
        days[:,1] = np.arange(0,len(days[:,0]))
        mu = np.exp(days.dot(beta[:2]))
        od = beta[2]*np.ones(len(days[:,1]))
        r=mu/(od-1)
        p = 1/od
        data = stats.nbinom.rvs(r,p)
        print('data:', data)

        negative_binomial_test = covid_models.negative_binomial_data(data)
        number_of_samples = 100000
        initial_position = np.array([ 1.13897153,  0.3940096 , 56.45188604])
        step_size = 0.00005
        # proposal_covariance = np.array([[ 1.99563291e-02, -1.18693670e-03,  5.86064693e-04],
        #                                 [-1.18693670e-03,  8.29099377e-05, -2.06102988e-06],
        #                                 [ 5.86064693e-04, -2.06102988e-06,  1.24682610e-01]])
        thinning_rate = 1

        output = covid_inference.mala(negative_binomial_test,
                                      number_of_samples,
                                      initial_position,
                                      step_size,
                                      thinning_rate=thinning_rate)

        proposal_covariance = np.cov(output.T)
        step_size = 3.0
        output =  covid_inference.mala(negative_binomial_test,
                                      number_of_samples,
                                      initial_position,
                                      step_size,
                                      proposal_covariance=proposal_covariance,
                                      thinning_rate=thinning_rate)

        # test that mean and variance inferences are within 0.01 of the ground truth
        # np.testing.assert_allclose(np.mean(output[:,0]),mean[0],0.1)
        # np.testing.assert_allclose(np.var(output[:,0]),covariance_matrix[0,0],0.1)

        # test that we get the expected number of samples
        # np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        # np.testing.assert_almost_equal(output.shape[1],len(initial_position))
        print('x_1 mean:',np.mean(output[:,0]))
        print('x_2 mean:',np.mean(output[:,1]))
        print('x_3 mean:',np.mean(output[:,2]))

        plt.clf()
        fig, ax = plt.subplots(3,1,figsize=(10,10))
        for i in range(output.shape[1]):
            ax[i].plot(output[:,i])
            ax[i].set_xlabel('$\\beta_{}$'.format(i))
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','traceplots_mala_negative_binomial.pdf'))
        #
        # # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output[:-1:10],columns=['$x_1$','$x_2$','$x_3$']),diag_sharey=False)
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_mala_negative_binomial.pdf'))

    def xest_sm_mala_on_negative_binomial(self):
        print("\nTesting smMALA on a negative binomial distribution with data")

        # make some data
        n = 30
        beta = np.array([1,0.4,100])
        days = np.zeros((n,2))
        days[:,0] = 1
        days[:,1] = np.arange(0,len(days[:,0]))
        mu = np.exp(days.dot(beta[:2]))
        od = beta[2]*np.ones(len(days[:,1]))
        r=mu/(od-1)
        p = 1/od
        data = stats.nbinom.rvs(r,p)
        print('data:', data)

        negative_binomial_test = covid_models.negative_binomial_data(data)
        number_of_samples = 100000
        # initial_position = np.array([ 1.13897153,  0.3940096 , 8.45188604])
        initial_position = beta
        step_size = 0.4
        # proposal_covariance = np.array([[ 1.99563291e-02, -1.18693670e-03,  5.86064693e-04],
        #                                 [-1.18693670e-03,  8.29099377e-05, -2.06102988e-06],
        #                                 [ 5.86064693e-04, -2.06102988e-06,  1.24682610e-01]])
        thinning_rate = 1

        output = covid_inference.simplified_manifold_mala(negative_binomial_test,
                                                      number_of_samples,
                                                      initial_position,
                                                      step_size,
                                                      thinning_rate=thinning_rate)

        # proposal_covariance
        # test that mean and variance inferences are within 0.01 of the ground truth
        # np.testing.assert_allclose(np.mean(output[:,0]),mean[0],0.1)
        # np.testing.assert_allclose(np.var(output[:,0]),covariance_matrix[0,0],0.1)

        # test that we get the expected number of samples
        # np.testing.assert_almost_equal(output.shape[0],number_of_samples)
        # np.testing.assert_almost_equal(output.shape[1],len(initial_position))
        print('x_1 mean:',np.mean(output[:,0]))
        print('x_2 mean:',np.mean(output[:,1]))
        print('x_3 mean:',np.mean(output[:,2]))

        plt.clf()
        fig, ax = plt.subplots(3,1,figsize=(10,10))
        for i in range(output.shape[1]):
            ax[i].plot(output[:,i])
            ax[i].set_xlabel('$\\beta_{}$'.format(i))
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','traceplots_sm_mala_negative_binomial.pdf'))
        #
        # # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output[:-1:10],columns=['$x_1$','$x_2$','$x_3$']),diag_sharey=False)
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True)
        g = g.map_diag(sns.distplot,norm_hist=True,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_sm_mala_negative_binomial.pdf'))


    def test_delayed_model_class(self):
        print("\nTesting random walk on Katrina delay model")
        data = np.genfromtxt('datafit_EN.csv', delimiter=",")[:,0]
        delayed_model = covid_models.delayed_compartment_model(data)

        number_of_samples = 50000
        initial_position = np.array([0.7,0.4,0.1,2.0,10.0])
        step_size = 0.0032
        proposal_covariance = np.diag(np.array([1.0,1.0,1.0,5.0,100.0]))
        thinning_rate = 1

        output = covid_inference.random_walk(delayed_model,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             proposal_covariance=proposal_covariance,
                                             thinning_rate=thinning_rate)

        proposal_covariance = np.cov(output[number_of_samples//2:].T)
        initial_position = np.mean(output[number_of_samples//2:],axis=0)
        print('initial position',initial_position)
        print('covariance matrix',proposal_covariance)

        step_size = 1.8
        output = covid_inference.random_walk(delayed_model,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             proposal_covariance=proposal_covariance,
                                             thinning_rate=thinning_rate)

        print(np.mean(output,axis=0))

        plt.clf()
        fig, ax = plt.subplots(5,1,figsize=(10,10))
        for i in range(output.shape[1]):
            ax[i].plot(output[:,i])
            ax[i].set_xlabel('$\\theta_{}$'.format(i))
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','traceplots_random_walk_katrina.pdf'))
        # plot a pairgrid
        plt.clf()
        g = sns.PairGrid(pd.DataFrame(output,columns=['$\\theta_1$','$\\theta_2$','$\\theta_3$','$\\theta_4$','$\\theta_5$']),diag_sharey=False)
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_random_walk_katrina.pdf'))
