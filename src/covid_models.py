import numpy as np
import scipy.stats as st
from scipy.special import factorial, polygamma

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
        if np.any(position < 0):
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
        pass
        # return log_likelihood_hessian

class poisson_data:
    """

    """
    def __init__(self,data,days):
        """
        Constructor function for the poisson class for data.

        Parameters
        ----------

        data : numpy array
            a 1xn array containing number of new cases per day for a region. The first column is days, second column
            is the cases.

        days : numpy array
            a 2xn array containing the days. The first row are all 1's, the second row are the days in order.

        """
        self.data = data
        self.days = days

    def log_likelihood(self, position):
        """
        A function returning the log likelihood of the poisson data distribution for a given position in a
        Markov chain.

        Parameters
        ----------

        position : numpy array
            a 3x1 numpy array giving the current position in the Markov chain.

        Returns
        -------

        log_likelihood : double
            the log of the hybrid rosenbrock likelihood function.

        """
        if np.any(position < 0):
            return -np.inf
        else:
            # day_numbers = np.arange(0,len(self.data))
            # mean = position[0]*np.exp(position[1]*day_numbers)
            # r = mean/(variance-1)
            # p = 1/variance
            log_likelihood = np.sum(self.data*(position.dot(self.days.T)) - np.exp(position.dot(self.days.T)) - np.log(factorial(self.data)))
            return log_likelihood

    def log_likelihood_gradient(self,position):
        """
        A function returning the gradient of the log likelihood of the poisson data distribution
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
        pass
        # day_numbers = np.arange(0,len(data))
        # mean = position[0]*np.exp(position[1]*day_numbers)
        # variance = position[2]*np.ones(len(day_numbers))
        # r = mean/(variance-1)
        # p = 1/variance
        # log_likelihood = np.sum(st.nbinom.logpmf(self.data,r,p))
        # return log_likelihood_gradient

    def log_likelihood_hessian(self,position):
        """
        A function returning the Hessian of the log likelihood of the poisson data distribution
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
        pass
        # return log_likelihood_hessian
