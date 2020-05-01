import numpy as np
import scipy.stats as st
from scipy.special import factorial, polygamma
from scipy.integrate import solve_ivp
import numba
from numba import jitclass, float64, int64
import sys

class normal:
    """
    A one dimensional normal distribution with user specified mean and variance. Can be used as a
    target distribution in MCMC methods.
    """
    def __init__(self,mean,variance):
        """
        Constructor function for the normal class.

        Parameters
        ----------

        mean : double
            the mean of the normal distribution

        variance : double
            the variance of the normal distribution
        """
        self.mean = mean
        self.variance = variance

    def log_likelihood(self, position):
        """
        A function returning the log likelihood of the normal distribution for a given position in a
        Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood : double
            the log of the normal likelihood function.

        """
        log_likelihood = -np.power((position-self.mean),2)/(2*self.variance)
        return log_likelihood

    def log_likelihood_gradient(self,position):
        """
        A function returning the gradient of the log likelihood of the normal distribution
        for a given position in a Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood_gradient : double
            the gradient of the log of the normal likelihood function.

        """
        log_likelihood_gradient = -(position-self.mean)/(self.variance)
        return log_likelihood_gradient

    def log_likelihood_hessian(self,position):
        """
        A function returning the Hessian of the log likelihood of the normal distribution
        for a given position in a Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood_hessian : double
            the Hessian of the log of the normal likelihood function.

        """
        log_likelihood_hessian = -1/self.variance
        return log_likelihood_hessian

class multivariate_normal:
    """
    A multi dimensional normal distribution with user specified mean and symmetric, positive definite,
    covariance matrix. Can be used as a target distribution in MCMC methods.
    """
    def __init__(self,mean,covariance_matrix):
        """
        Constructor function for the multivariate normal class. This computes the covariance matrix inverse for
        use in the subsequent methods.

        Parameters
        ----------

        mean : numpy array
            the mean vector of the multivariate normal distribution

        covariance_matrix : numpy array
            the covariance matrix of the multivariate normal distribution
        """
        self.mean = mean
        self.covariance_matrix = covariance_matrix
        self.covariance_matrix_inverse = np.linalg.inv(covariance_matrix)

    def log_likelihood(self, position):
        """
        A function returning the log likelihood of the multivariate normal distribution for a given position in a
        Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood : double
            the log of the multivariate normal likelihood function.

        """
        log_likelihood = -0.5*np.transpose(position-self.mean).dot(self.covariance_matrix_inverse).dot(position-self.mean)
        return log_likelihood

    def log_likelihood_gradient(self,position):
        """
        A function returning the gradient of the log likelihood of the multivariate normal distribution
        for a given position in a Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood_gradient : double
            the gradient of the log of the multivariate normal likelihood function.

        """
        log_likelihood_gradient = -self.covariance_matrix_inverse.dot(position-self.mean)
        return log_likelihood_gradient

    def log_likelihood_hessian(self,position):
        """
        A function returning the Hessian of the log likelihood of the multivariate normal distribution
        for a given position in a Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood_hessian : double
            the Hessian of the log of the multivariate normal likelihood function.

        """
        log_likelihood_hessian = -self.covariance_matrix_inverse
        return log_likelihood_hessian

class hybrid_rosenbrock23:
    """
    A multi dimensional distribution, detailed by Pagani (2019). See https://arxiv.org/abs/1903.09556 .
    Here n_1 = 3 and n_2 = 2. Can be used as a target distribution in MCMC methods.
    """
    def __init__(self,mean,shape):
        """
        Constructor function for the hybrid rosenbrock class.

        Parameters
        ----------

        mean : double
            the mean, mu, of the hybrid rosenbrock distribution

        shape : numpy array
            a 1x5 vector containing one half the precision of each normal kernel

        """
        self.mean = mean
        self.shape = shape

    def log_likelihood(self, position):
        """
        A function returning the log likelihood of the hybrid rosenbrock distribution for a given position in a
        Markov chain.

        Parameters
        ----------

        position : numpy array
            a 5x1 numpy array giving the current position in the Markov chain.

        Returns
        -------

        log_likelihood : double
            the log of the hybrid rosenbrock likelihood function.

        """
        log_likelihood = -(self.shape[0]*np.power(position[0]-self.mean,2) +
                           self.shape[1]*np.power(position[1]-np.power(position[0],2),2) +
                           self.shape[2]*np.power(position[2]-np.power(position[1],2),2) +
                           self.shape[3]*np.power(position[3]-np.power(position[0],2),2) +
                           self.shape[4]*np.power(position[4]-np.power(position[3],2),2))
        return log_likelihood

    def log_likelihood_gradient(self,position):
        """
        A function returning the gradient of the log likelihood of the hybrid rosenbrock distribution
        for a given position in a Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood_gradient : double
            the gradient of the log of the multivariate normal likelihood function.

        """
        log_likelihood_gradient = np.array([2*self.shape[0]*self.mean - 8*self.shape[1]*position[0]**3 + position[0]*(-2*self.shape[0] + 4*self.shape[1]*position[1] +4*self.shape[1]*position[3]),
                                            2*self.shape[1]*(position[0]**2 +position[1]*(-1 -2*position[1]**2 +2*position[2]) ),
                                            2*self.shape[2]*(position[1]**2 -position[2]),
                                            2*self.shape[3]*(position[0]**2 +position[3]*(-1 -2*position[3]**2 +2*position[4]) ),
                                            2*self.shape[4]*(position[3]**2 -position[4])])
        return log_likelihood_gradient

    def log_likelihood_hessian(self,position):
        """
        A function returning the Hessian of the log likelihood of the multivariate normal distribution
        for a given position in a Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood_hessian : double
            the Hessian of the log of the multivariate normal likelihood function.

        """
        log_likelihood_hessian = np.array([[-2*self.shape[0] - 24*self.shape[1]*position[0]**2 + 4*self.shape[1]*position[1]+4*self.shape[1]*position[3],
                                             4*self.shape[1]*position[0],0,4*self.shape[1]*position[0],0],
                                            [4*self.shape[1]*position[0],-2*self.shape[1]*(1 + 6*position[1]**2 - 2*position[2]),4*self.shape[1]*position[1],0,0],
                                            [0,4*self.shape[1]*position[1],-2*self.shape[1],0,0],
                                            [4*self.shape[1]*position[0],0,0,-2*self.shape[1]*(1 + 6*position[3]**2 - 2*position[4]),4*self.shape[1]*position[3]],
                                            [0,0,0,4*self.shape[1]*position[3],-2*self.shape[1]]])
        return log_likelihood_hessian

class negative_binomial_data:
    """

    """
    def __init__(self,data):
        """
        Constructor function for the negative binomial class for data.

        Parameters
        ----------

        data : numpy array
            a 1xn array containing number of new cases per day for a region

        """
        self.data = data

    def log_likelihood(self, position,negative=False):
        """
        A function returning the log likelihood of the negative binomial distribution for a given position in a
        Markov chain.

        Parameters
        ----------

        position : numpy array
            a 3x1 numpy array giving the current position in the Markov chain.

        negative : bool
            default value is False. When set to True, log_likelihood returns the negative log likelihood.

        Returns
        -------

        log_likelihood : double
            the log of the negative binomial likelihood function (or the negative of this, when 'negative' is
            set to True).

        """
        if np.any(position < 0) or position[2] < 1:
            if negative==True:
                return np.inf
            return -np.inf
        else:
            day_numbers = np.arange(len(self.data))
            mean = np.exp(position[0])*np.exp(position[1]*day_numbers)
            overdispersion = position[2]*np.ones(len(day_numbers))
            r = mean/(overdispersion-1)
            p = 1/overdispersion
            log_likelihood = np.sum(st.nbinom.logpmf(self.data,r,p))
            if negative==True:
                return -log_likelihood
            return log_likelihood

    def log_likelihood_gradient(self,position):
        """
        A function returning the gradient of the log likelihood of the negative binomial distribution
        for a given position in a Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood_gradient : double
            the gradient of the log of the multivariate normal likelihood function.

        """
        day_numbers = np.arange(len(self.data))

        dx = np.sum((np.exp(position[0] + position[1]*day_numbers)*(np.log(1/position[2]) +
                    polygamma(0,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                    polygamma(0,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)))) / (
                    (position[2] - 1) ))

        dy = np.sum((day_numbers*np.exp(position[0] + position[1]*day_numbers)*(np.log(1/position[2]) +
                    polygamma(0,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                    polygamma(0,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)))) / (
                    (position[2] - 1) ))

        dz = np.sum((self.data*(position[2]-1) - np.exp(position[0] + position[1]*day_numbers)*(position[2]-1 + position[2]*np.log(1/position[2]) +
                    position[2]*polygamma(0,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                    position[2]*polygamma(0,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)))) / (
                    np.power((position[2]-1),2)*position[2] ))

        log_likelihood_gradient = np.array([dx,dy,dz])

        return log_likelihood_gradient

    def log_likelihood_hessian(self,position):
        """
        A function returning the Hessian of the log likelihood of the negative binomial distribution
        for a given position in a Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood_hessian : double
            the Hessian of the log of the multivariate normal likelihood function.

        """
        day_numbers = np.arange(len(self.data))

        dxdx = np.sum((np.exp(position[0] + position[1]*day_numbers)*((position[2]-1)*(np.log(1/position[2]) +
                       polygamma(0,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                       polygamma(0,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1))) +
                       np.exp(position[0] + position[1]*day_numbers)*(polygamma(1,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                       polygamma(1,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1))))) / (
                       np.power((position[2] - 1),2) ))

        dydy = np.sum((np.exp(position[0] + position[1]*day_numbers)*np.power(day_numbers,2)*((position[2]-1)*(np.log(1/position[2]) +
                       polygamma(0,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                       polygamma(0,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1))) +
                       np.exp(position[0] + position[1]*day_numbers)*(polygamma(1,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                       polygamma(1,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1))))) / (
                       np.power((position[2] - 1),2) ))

        dzdz  = np.sum((-self.data*np.power((position[2]-1),2)*(2*position[2]-1) +
                        (position[2]-1)*np.exp(position[0] + position[1]*day_numbers)*(1-4*position[2]+3*np.power(position[2],2) +
                         2*np.power(position[2],2)*np.log(1/position[2]) + 2*np.power(position[2],2)*(polygamma(0,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                         polygamma(0,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)))) +
                         np.power(position[2],2)*np.exp(2*(position[0] + position[1]*day_numbers))*(polygamma(1,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                         polygamma(1,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)))) / (
                         np.power((position[2]-1),4)*np.power(position[2],2) ))

        dxdy = np.sum((np.exp(position[0] + position[1]*day_numbers)*day_numbers*((position[2]-1)*(np.log(1/position[2]) +
                       polygamma(0,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                       polygamma(0,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1))) +
                       np.exp(position[0] + position[1]*day_numbers)*(polygamma(1,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                       polygamma(1,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1))))) / (
                       np.power((position[2] - 1),2) ))

        dxdz = np.sum((np.exp(position[0] + position[1]*day_numbers)*(-1*(position[2]-1)*(position[2]-1+position[2]*np.log(1/position[2]) +
                       position[2]*polygamma(0,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                       position[2]*polygamma(0,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1))) -
                       position[2]*np.exp(position[0] + position[1]*day_numbers)*(polygamma(1,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                       polygamma(1,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1))))) / (
                       position[2]*np.power((position[2] - 1),3) ))

        dydz = np.sum((np.exp(position[0] + position[1]*day_numbers)*day_numbers*(-1*(position[2]-1)*(position[2]-1+position[2]*np.log(1/position[2]) +
                       position[2]*polygamma(0,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                       position[2]*polygamma(0,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1))) -
                       position[2]*np.exp(position[0] + position[1]*day_numbers)*(polygamma(1,self.data + np.exp(position[0] + position[1]*day_numbers)/(position[2]-1)) -
                       polygamma(1,np.exp(position[0] + position[1]*day_numbers)/(position[2]-1))))) / (
                       position[2]*np.power((position[2] - 1),3) ))

        log_likelihood_hessian = np.array([[dxdx,dxdy,dxdz],
                                           [dxdy,dydy,dydz],
                                           [dxdz,dydz,dzdz]])

        return log_likelihood_hessian

class delayed_compartment_model:
    """
    # TODO:
    """
    def __init__(self,region='EN',fit=['hospital_incidence']):
        """
        Constructor function for the delayed_compartment_model class.

        Parameters
        ----------

        region : str
            A two letter string representing the region of the UK you want to make
            predictions for. Options are:
            - 'EN', England
            - 'EE', East England
            - 'LO', London
            - 'MI', Midlands
            - 'NE', North East
            - 'NW', North West
            - 'SE', South East
            - 'SW', South West

        fit : list[str]
            A list containing strings which determine which data to fit. The list can contain
            any combination of the options. Options are:
            - 'hospital_incidence'
            - 'hospital_prevalence'
            - 'icu_prevalence'
            - 'death_incidence'
        """
        self.region = region
        self.fit = fit
        self.region_specific_initialisation()
        self.number_of_states = 21
        self.control_dates = np.array([10,43,48,89])

        alternative_parameters = 0
        # set rates
        Ea = 3
        rE1 = 1/5.5
        rE = rE1*Ea
        IHa = 2
        rIH1 = 1/5
        rIH = rIH1*IHa
        rIR = 1/3.5
        if alternative_parameters:
            rCD = 1/9
            rHC = 1/0.5
            rHR = 1/7.5
            rCR = 1/12
        else:
            rHC = 1/2.05
            rHR = 1/10
            rCD = 1/11.1
            rCM = 1/10.55
            rMR = 1/5.73
        rA = 1/3.5
        rX = 1/4.5
        self.rates = np.array([rE,rIH,rIR,rHC,rHR,rCD,rCM,rMR,rA,rX])

        # set proportions
        pA = 0.18
        pH = 0.15
        if alternative_parameters:
            pC = 0.16
            pD = 0.68
        else:
            pC = 0.16
            pD = 0.68
            pU = 0.37
            pT = (1-pU)/pU * pC/(1-pC) * pD

        self.probabilities = np.array([pA,pH,pC,pD,pT])
        # set transmission parameters
        R0 = 3.62
        f = 0.25
        k = (1-pA)*(f/rE + pH/rIH + (1-pH)/rIR) + pA*f*(1/rE + 1/rA)
        b = R0/k
        h0 = 0.1
        self.transmission_rates = np.array([b,f,h0])

        # Initial conditions
        log_initial_infectious = np.log(0.1)
        self.initial_state = np.array([self.initial_population-np.exp(log_initial_infectious),(1-pA)*np.exp(log_initial_infectious),
                                       0,0,0,0,0,0,0,0,0,0,pA*np.exp(log_initial_infectious),0,0,0,0,0,0,self.initial_population,0])

    def region_specific_initialisation(self):
        """
        Imports the correct data based on region input
        """
        if self.region == 'EN':
            # set data
            data = np.genfromtxt('../test/data/datafit_EN.csv', delimiter=",")[:,0:3]
            self.hospital_incidence_data  = data[:,0]
            self.hospital_prevalence_data = data[:,1]
            self.icu_prevalence_data      = data[:,2]
            self.death_data = np.genfromtxt('../test/data/datafit_deaths.csv', delimiter=",")[:,7]
            # construct indices for log likelihood
            self.data_first_day = 37
            self.hospital_incidence_indices  = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_incidence_data))
            self.hospital_prevalence_indices = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_prevalence_data))
            self.icu_prevalence_indices      = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.icu_prevalence_data))
            self.death_indices               = np.arange(46,
                                                         46+len(self.death_data))
            self.initial_population = 56000000

        elif self.region == 'EE':
            # set data
            data = np.genfromtxt('../test/data/datafit_regions.csv', delimiter=",")[:,0:3]
            self.hospital_incidence_data  = data[:,0]
            self.hospital_prevalence_data = data[:,1]
            self.icu_prevalence_data      = data[:,2]
            self.death_data = np.genfromtxt('../test/data/datafit_deaths.csv', delimiter=",")[:,0]
            self.data_first_day = 46
            self.hospital_incidence_indices  = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_incidence_data))
            self.hospital_prevalence_indices = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_prevalence_data))
            self.icu_prevalence_indices      = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.icu_prevalence_data))
            self.death_indices               = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.death_data))
            self.initial_population = 6200000

        elif self.region == 'LO':
            data = np.genfromtxt('../test/data/datafit_regions.csv', delimiter=",")[:,2:5]
            self.hospital_incidence_data  = data[:,0]
            self.hospital_prevalence_data = data[:,1]
            self.icu_prevalence_data      = data[:,2]
            self.death_data = np.genfromtxt('../test/data/datafit_deaths.csv', delimiter=",")[:,1]
            self.data_first_day = 46
            self.hospital_incidence_indices  = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_incidence_data))
            self.hospital_prevalence_indices = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_prevalence_data))
            self.icu_prevalence_indices      = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.icu_prevalence_data))
            self.death_indices               = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.death_data))
            self.initial_population = 8900000

        elif self.region == 'MI':
            data = np.genfromtxt('../test/data/datafit_regions.csv', delimiter=",")[:,5:8]
            self.hospital_incidence_data  = data[:,0]
            self.hospital_prevalence_data = data[:,1]
            self.icu_prevalence_data      = data[:,2]
            self.death_data = np.genfromtxt('../test/data/datafit_deaths.csv', delimiter=",")[:,2]
            self.data_first_day = 46
            self.hospital_incidence_indices  = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_incidence_data))
            self.hospital_prevalence_indices = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_prevalence_data))
            self.icu_prevalence_indices      = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.icu_prevalence_data))
            self.death_indices               = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.death_data))
            self.initial_population = 10700000

        elif self.region == 'NE':
            data = np.genfromtxt('../test/data/datafit_regions.csv', delimiter=",")[:,8:11]
            self.hospital_incidence_data  = data[:,0]
            self.hospital_prevalence_data = data[:,1]
            self.icu_prevalence_data      = data[:,2]
            self.death_data = np.genfromtxt('../test/data/datafit_deaths.csv', delimiter=",")[:,3]
            self.data_first_day = 46
            self.hospital_incidence_indices  = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_incidence_data))
            self.hospital_prevalence_indices = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_prevalence_data))
            self.icu_prevalence_indices      = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.icu_prevalence_data))
            self.death_indices               = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.death_data))
            self.initial_population = 3210000

        elif self.region == 'NW':
            data = np.genfromtxt('../test/data/datafit_regions.csv', delimiter=",")[:,11:14]
            self.hospital_incidence_data  = data[:,0]
            self.hospital_prevalence_data = data[:,1]
            self.icu_prevalence_data      = data[:,2]
            self.death_data = np.genfromtxt('../test/data/datafit_deaths.csv', delimiter=",")[:,4]
            self.data_first_day = 46
            self.hospital_incidence_indices  = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_incidence_data))
            self.hospital_prevalence_indices = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_prevalence_data))
            self.icu_prevalence_indices      = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.icu_prevalence_data))
            self.death_indices               = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.death_data))
            self.initial_population = 7300000

        elif self.region == 'SE':
            data = np.genfromtxt('../test/data/datafit_regions.csv', delimiter=",")[:,14:17]
            self.hospital_incidence_data  = data[:,0]
            self.hospital_prevalence_data = data[:,1]
            self.icu_prevalence_data      = data[:,2]
            self.death_data = np.genfromtxt('../test/data/datafit_deaths.csv', delimiter=",")[:,5]
            self.data_first_day = 46
            self.hospital_incidence_indices  = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_incidence_data))
            self.hospital_prevalence_indices = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_prevalence_data))
            self.icu_prevalence_indices      = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.icu_prevalence_data))
            self.death_indices               = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.death_data))
            self.initial_population = 9130000

        elif self.region == 'SW':
            data = np.genfromtxt('../test/data/datafit_regions.csv', delimiter=",")[:,17:20]
            self.hospital_incidence_data  = data[:,0]
            self.hospital_prevalence_data = data[:,1]
            self.icu_prevalence_data      = data[:,2]
            self.death_data = np.genfromtxt('../test/data/datafit_deaths.csv', delimiter=",")[:,6]
            self.data_first_day = 46
            self.hospital_incidence_indices  = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_incidence_data))
            self.hospital_prevalence_indices = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.hospital_prevalence_data))
            self.icu_prevalence_indices      = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.icu_prevalence_data))
            self.death_indices               = np.arange(self.data_first_day,
                                                         self.data_first_day+len(self.death_data))
            self.initial_population = 5600000

        else:
            print("Invalid region!")
            sys.exit(1)

    def delayed_ode(self, t, y, rates, probabilities, transmission_rates):
        """
        The ODE model from Katrina Lythgoe, Lorenzo Pellis et al. (2020). It describes a compartmental
        SEIR model, with further sub classification of hospitalised and intensive care patients.

        Parameters
        ----------

        t : list
            A list with two entries which defines the time range over which the ODE is integrated.
            The first entry is the start time, the second entry is the end time.

        y : numpy array
            An array defining the states of the model. Some are repeated for a delayed component. They are
            as follows:
            - y[0]:  S
            - y[1]:  ES
            - y[4]:  IH
            - y[6]:  IR
            - y[7]:  HC
            - y[8]:  HR
            - y[9]:  CD
            - y[10]: CR
            - y[11]: M
            - y[12]: EA
            - y[15]: IA
            - y[16]: R
            - y[17]: D
            - y[18]: X
            - y[19]: N
            - y[20]: HD

        rates : numpy array
            An array defining the rate constants of the model. Some are repeated for a delayed component. They are
            as follows:
            - rates[0]: rE
            - rates[1]: rIH
            - rates[2]: rIR
            - rates[3]: rHC
            - rates[4]: rHR
            - rates[5]: rCD
            - rates[6]: rCM
            - rates[7]: rMR
            - rates[8]: rA
            - rates[9]: rX

        probabilities : numpy array
            An array defining the probabilities/proportions of the model. Some are
            repeated for a delayed component. They are as follows:
            - probabilities[0]: pA
            - probabilities[1]: pH
            - probabilities[2]: pC
            - probabilities[3]: pD
            - probabilities[4]: pT

        transmission_rates : numpy array
            An array defining the transmission rates of the model. Some are
            repeated for a delayed component. They are as follows:
            - transmission_rates[0]: b
            - transmission_rates[1]: f
            - transmission_rates[2]: h0

        Returns
        -------

        dydt : numpy array
            The value of the derivatives of each state variable, y, at time, t
        """
        dydt = np.array([ -y[0] * transmission_rates[0] * ( transmission_rates[1]*(y[1]+y[2]+y[3]+y[12]+y[13]+y[14]+y[15]) + (y[4]+y[5]+y[6]) ) / y[19],
                (1-probabilities[0]) * y[0] * transmission_rates[0] * ( transmission_rates[1]*(y[1]+y[2]+y[3]+y[12]+y[13]+y[14]+y[15]) + (y[4]+y[5]+y[6]) ) / y[19] - rates[0]*y[1],
                rates[0]*y[1] - rates[0]*y[2],
                rates[0]*y[2] - rates[0]*y[3],
                probabilities[1]*rates[0]*y[3] - rates[1]*y[4],
                rates[1]*y[4] - rates[1]*y[5],
                (1-probabilities[1])*rates[0]*y[3] - rates[2]*y[6],
                probabilities[2]*rates[1]*y[5] - rates[3]*y[7],
                (1-probabilities[2]-probabilities[4])*rates[1]*y[5] - rates[4]*y[8],
                probabilities[3]*rates[3]*y[7] - rates[5]*y[9],
                (1-probabilities[3])*rates[3]*y[7] - rates[6]*y[10],
                rates[6]*y[10] - rates[7]*y[11],
                probabilities[0] * y[0] * transmission_rates[0] * ( transmission_rates[1]*(y[1]+y[2]+y[3]+y[12]+y[13]+y[14]+y[15]) + (y[4]+y[5]+y[6]) ) / y[19] - rates[0]*y[12],
                rates[0]*y[12] - rates[0]*y[13],
                rates[0]*y[13] - rates[0]*y[14],
                rates[0]*y[14] - rates[8]*y[15],
                rates[2]*y[6] + rates[4]*y[8] + rates[7]*y[11] + rates[8]*y[15],
                rates[5]*y[9] + rates[4]*y[20] - rates[9]*y[17],
                rates[9]*y[17],
                -rates[5]*y[9] - rates[4]*y[20],
                probabilities[4]*rates[1]*y[5] - rates[4]*y[20]])

        return dydt

    def solve_ode(self,position,time_range=[0,99]):
        transmission_rates = np.copy(self.transmission_rates)
        rates = np.copy(self.rates)
        probabilities = np.copy(self.probabilities)
        control_dates = np.copy(self.control_dates)
        control_dates[-1] = time_range[-1]
        rCD = rates[5]
        beta = transmission_rates[0]
        reduced_beta = position[0:3]*beta
        log_initial_infectious = position[3]
        sigma_0 = position[4]
        rCM = position[5]
        rHR = position[6]
        pC = position[7]
        pT = position[8]
        rates[[4,6]] = np.array([rHR,rCM])
        probabilities[[2,4]] = np.array([pC,pT])

        # solve the ODE
        Yt = np.zeros((len(control_dates)+1,self.number_of_states))
        Y0 = np.array([self.initial_population-np.exp(log_initial_infectious),(1-probabilities[0])*np.exp(log_initial_infectious),
                       0,0,0,0,0,0,0,0,0,0,probabilities[0]*np.exp(log_initial_infectious),0,0,0,0,0,0,self.initial_population,0])
        Yt[0] = Y0
        sol = solve_ivp(self.delayed_ode,
                        [0,control_dates[0]],
                        Y0,
                        t_eval=np.arange(0,control_dates[0]+1),
                        args=(rates,probabilities,transmission_rates))
        Ttemp = sol.t
        Ytemp = sol.y
        Yt[1] = np.array([item[-1] for item in Ytemp])
        Tall = Ttemp
        Yall = Ytemp
        for ic in range(len(control_dates)-1):
            time_range = np.array([control_dates[ic],control_dates[ic+1]])
            transmission_rates[0] = reduced_beta[ic]
            sol = solve_ivp(self.delayed_ode,
                            time_range,
                            Yt[ic+1],
                            t_eval=np.linspace(control_dates[ic],control_dates[ic+1],int(control_dates[ic+1]-control_dates[ic])+1),
                            args=(rates,probabilities,transmission_rates))
            Ttemp = sol.t
            Ytemp = sol.y
            Yt[ic+2] = np.array([item[-1] for item in Ytemp])
            Tall = np.append(Tall,Ttemp[1:])
            Yall = np.append(Yall,np.array([item[1:] for item in Ytemp]),axis=-1)

        return Yall

    def log_likelihood(self, position):
        """
        A function returning the log target of the delayed_compartment_model for a given position in a
        Markov chain.

        Parameters
        ----------

        position : numpy array
            a 3x1 numpy array giving the current position in the Markov chain.

        Returns
        -------

        log_likelihood : double
            the log of the delayed_compartment_model target function.

        """
        number_of_parameters = len(position)
        if np.any(position[[0,1,2,4]] < 0):
            return -np.inf
        else:
            transmission_rates = np.copy(self.transmission_rates)
            rates = np.copy(self.rates)
            probabilities = np.copy(self.probabilities)
            rCD = rates[5]
            beta = transmission_rates[0]
            reduced_beta = position[0:3]*beta
            log_initial_infectious = position[3]
            sigma_0 = position[4]
            rCM = position[5]
            rHR = position[6]
            pC = position[7]
            pT = position[8]
            rates[[4,6]] = np.array([rHR,rCM])
            probabilities[[2,4]] = np.array([pC,pT])

            # solve the ODE
            Yt = np.zeros((len(self.control_dates)+1,self.number_of_states))
            Y0 = np.array([self.initial_population-np.exp(log_initial_infectious),(1-probabilities[0])*np.exp(log_initial_infectious),
                           0,0,0,0,0,0,0,0,0,0,probabilities[0]*np.exp(log_initial_infectious),0,0,0,0,0,0,self.initial_population,0])
            Yt[0] = Y0
            sol = solve_ivp(self.delayed_ode,
                            [0,self.control_dates[0]],
                            Y0,
                            t_eval=np.arange(0,self.control_dates[0]+1),
                            args=(rates,probabilities,transmission_rates))
            Ttemp = sol.t
            Ytemp = sol.y
            Yt[1] = np.array([item[-1] for item in Ytemp])
            Tall = Ttemp
            Yall = Ytemp
            for ic in range(len(self.control_dates)-1):
                time_range = np.array([self.control_dates[ic],self.control_dates[ic+1]])
                transmission_rates[0] = reduced_beta[ic]
                sol = solve_ivp(self.delayed_ode,
                                time_range,
                                Yt[ic+1],
                                t_eval=np.linspace(self.control_dates[ic],self.control_dates[ic+1],int(self.control_dates[ic+1]-self.control_dates[ic])+1),
                                args=(rates,probabilities,transmission_rates))
                Ttemp = sol.t
                Ytemp = sol.y
                Yt[ic+2] = np.array([item[-1] for item in Ytemp])
                Tall = np.append(Tall,Ttemp[1:])
                Yall = np.append(Yall,np.array([item[1:] for item in Ytemp]),axis=-1)

            # Calculate log likelihood given fitting specification
            log_likelihood = 0
            # import pdb; pdb.set_trace()
            if 'hospital_incidence' in self.fit:
                log_likelihood += np.sum(st.nbinom.logpmf(self.hospital_incidence_data,
                                                          self.rates[1]*Yall[5,self.hospital_incidence_indices]/(sigma_0-1),
                                                          1/sigma_0))

            if 'hospital_prevalence' in self.fit:
                log_likelihood += np.sum(st.nbinom.logpmf(self.hospital_prevalence_data,
                                                          np.sum(Yall[[7,8,9,10,11,20]],axis=0)[self.hospital_prevalence_indices]/(sigma_0-1),
                                                          1/sigma_0))

            if 'icu_prevalence' in self.fit:
                log_likelihood += np.sum(st.nbinom.logpmf(self.icu_prevalence_data,
                                                          np.sum(Yall[[9,10]],axis=0)[self.icu_prevalence_indices]/(sigma_0-1),
                                                          1/sigma_0))

            # import pdb; pdb.set_trace()

            if 'death_incidence' in self.fit:
                log_likelihood += np.sum(st.nbinom.logpmf(self.death_data,
                                                          (rHR*Yall[20,self.death_indices] + rCD*Yall[9,self.death_indices])/(sigma_0-1),
                                                          1/sigma_0))
            return log_likelihood

    def log_likelihood_gradient(self,position):
        """
        A function returning the gradient of the log target of the delayed_compartment_model
        for a given position in a Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood_gradient : double
            the gradient of the log of the target.

        """
        pass
    def log_likelihood_hessian(self,position):
        """
        A function returning the Hessian of the log target of the delayed_compartment_model
        for a given position in a Markov chain.

        Parameters
        ----------

        position : numpy array
            the current position in the Markov chain.

        Returns
        -------

        log_likelihood_hessian : double
            the Hessian of the log of the target function.

        """
        pass
