import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import covid_inference
import covid_models
import os.path
import sys
import pandas as pd
import seaborn as sns; sns.set(); sns.set_style("ticks", {'axes.spines.right': False,
                                                          'axes.spines.top': False})
import matplotlib as mpl
# mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp
import multiprocessing as mp
import multiprocessing.pool as mp_pool
from datetime import datetime
startTime = datetime.now()
# make sure we find the right python module

delayed_model = covid_models.delayed_compartment_model(fit=['hospital_prevalence','hospital_incidence','icu_prevalence','death_incidence'])
saving_path = os.path.join(os.path.dirname(__file__), 'output','')

number_of_samples = 1000
initial_position = np.array([0.7,0.4,0.1,2.0,10.0,0.5,0.5,0.5,0.5])
step_size = 0.001
proposal_covariance = np.diag(np.array([1.0,1.0,1.0,5.0,100.0,1.0,1.0,1.0,1.0]))
thinning_rate = 1
        #
output = covid_inference.random_walk(delayed_model,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             proposal_covariance=proposal_covariance,
                                             thinning_rate=thinning_rate)
#print(output)
print(datetime.now() - startTime)
