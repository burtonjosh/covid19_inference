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

    def test_random_walk(self):
        normal_5_2 = covid_inference.normal(5,2)
        output = covid_inference.random_walk(normal_5_2,100000,np.array([1]),3)

        # plt.hist(output)
        # plt.show()

        print('mean',np.mean(output))
        print('variance',np.var(output))
