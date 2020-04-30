import numpy as np
import scipy.stats as st
from scipy.special import factorial, polygamma
from scipy.integrate import solve_ivp
import numba
from numba import jitclass, float64, int64

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

# @jitclass([('data', float64[:]),
#            ('number_of_states',int64),
#            ('control_dates', int64[:]),
#            ('rates', float64[:]),
#            ('probabilities', float64[:]),
#            ('transmission_rates', float64[:]),
#            ('initial_state', float64[:])])
class delayed_compartment_model:
    """
    # TODO:
    """
    def __init__(self,data):
        """
        Constructor function for the delayed_compartment_model class.

        Parameters
        ----------
# TODO:
        data : numpy array
            an nxm array.

        """
        self.data = data
        self.number_of_states = 21
        self.control_dates = np.array([10,43,48,99])

        fake = 0
        # Parameters
        # Rates:
        Ea = 3
        rE1 = 1/5.5 # Seems to be an assumption from Ferguson
        rE = rE1*Ea
        IHa = 2
        rIH1 = 1/5 # Katrina 5 days (seems again an assumption from Ferguson), Chris ~5-6 days
        rIH = rIH1*IHa
        rIR = 1/3.5#1/3.5
        if fake:
            rCD = 1/9#1/9 # Katrina 7 days, Chris ~8.8-9.5 days
            rHC = 1/0.5#1.65#5 # Katrina 5 days, Chris ~2 days *** This is to fit ***
            rHR = 1/7.5 # Katrina 12 days, Chris ~4.5 days
            rCR = 1/12 # Katrina 17 days, Chris ~12 days (but asking for sensitivity analysis 12-17 days
        else:
            rHC = 1/2.05 # Katrina 5 days, Chris ~2 days
        #     rHR = 1/12.18 # Katrina 12 days, Chris ~4.5 days
        #     rHD = 1/11.38 # Chris, from CHESS 11.38
            rHR = 1/10 # 10.91 to R (prob 1-pT) and 9.13 to D (prob pT)
        #     rCH = 1/15.49 # ICU cases going back to hospital
            rCD = 1/11.1 # Assuming Exponential (Gamma: mu = 8.97, sd = 7.37)
            rCM = 1/10.55 # M for monitoring
            rMR = 1/5.73
        #     CRa = 2
        #     rCR1 = 1/12.62 # Chris, from CHESS 12.62
        #     rCR = rCR1*CRa
        rA = 1/3.5
        rX = 1/4.5 # Rate at which deaths are registered (4-5 days from Nick Gents email with graph)
        self.rates = np.array([rE,rIH,rIR,rHC,rHR,rCD,rCM,rMR,rA,rX])
        # Proportions:
        pA = 0.18 # Katrina 0.179, Helena 0.25 from (Gostic, Gomez, Mummah, Kucharski, & Lloyd-Smith, 2020)
        pH = 0.15
        if fake:
            pC = 0.16#0.23 # Katrina 0.23, Jon Read 0.16 (ventilator need proxy for ICU)
            pD = 0.68#0.48 # Katrina 0.48, Jon Read 0.68 (ventilator need proxy for ICU)
        else:
            pC = 0.16
            pD = 0.68
            pU = 0.37 # Fraction of those that ultimately die who pass via ICU (37#)
            pT = (1-pU)/pU * pC/(1-pC) * pD # Formula su foglietto...

        self.probabilities = np.array([pA,pH,pC,pD,pT])
        # Transmission parameters:
        T2 = 2.1 # Katrina 2.1 (1.5, 3.7), Helena says from (Liu et al., 2020 Read, Bridgen, Cummings, Ho, & Jewell, 2020)
        R0 = 3.62 # Katrina 3.62 (2.43, 4.90)
        f = 0.25 # Fraction of baseline transmission if asymptomatic
        k = (1-pA) * ( f/rE + pH/rIH + (1-pH)/rIR ) + pA*f * ( 1/rE + 1/rA )
        b = R0/k
        h0 = 0.1
        self.transmission_rates = np.array([b,f,h0])
        indincnewhosp = 5
        indincnewdeath1 = 20
        indincnewdeath2 = 9
        indincnewicu = 7
        indprevhosp = [7,8,9,10,11,20]
        indprevicu = [9,10]

        # Initial conditions
        # J0 = 3 # number of initial infectives, seeded in ES and EA better stuff should be done
        # N0 = 1000000 # Maybe it doesn't matter
        log_initial_infectious = np.log(0.1) # 3*56 # number of initial infectives, seeded in ES and EA better stuff should be done
        N0 = 56000000 # Maybe it doesn't matter
        # initial_state = [ N0-J0, (1-pA)*J0, 0, 0, 0, 0, 0, 0, 0, 0, 0, pA*J0, 0, 0, 0, 0, N0 ]
        self.initial_state = np.array([N0-np.exp(log_initial_infectious),(1-pA)*np.exp(log_initial_infectious),
                                       0,0,0,0,0,0,0,0,0,0,pA*np.exp(log_initial_infectious),0,0,0,0,0,0,N0,0])
        # self.initial_state[[0,1,12,19]] = [N0-J0,(1-pA)*J0,pA*J0,N0]

    def delayed_ode(self, t, y, rates, probabilities, transmission_rates):
        # State variables, y: 1-4: S, ES, IH, IR; 5-8: HC, HR, CD, CR; 9-12: EA, IA, R, D; 13-14: X, N
        # Rates: 1-5: rE, rIH, rIR, rHC, rHR; 6-10: rCD, rCM, rMR, rA, rX
        # Probabilities: pA, pH, pC, pD, pT
        # transmission_rates: b (beta), f (alpha)
        dydt = np.array([ -y[0] * transmission_rates[0] * ( transmission_rates[1]*(y[1]+y[2]+y[3]+y[12]+y[13]+y[14]+y[15]) + (y[4]+y[5]+y[6]) ) / y[19], # S, y[0]
                (1-probabilities[0]) * y[0] * transmission_rates[0] * ( transmission_rates[1]*(y[1]+y[2]+y[3]+y[12]+y[13]+y[14]+y[15]) + (y[4]+y[5]+y[6]) ) / y[19] - rates[0]*y[1], # ES, y[1]
                rates[0]*y[1] - rates[0]*y[2],
                rates[0]*y[2] - rates[0]*y[3],
                probabilities[1]*rates[0]*y[3] - rates[1]*y[4], # IH, y[4]
                rates[1]*y[4] - rates[1]*y[5],
                (1-probabilities[1])*rates[0]*y[3] - rates[2]*y[6], # IR, y[6]
                probabilities[2]*rates[1]*y[5] - rates[3]*y[7], # HC, y[7]
                (1-probabilities[2]-probabilities[4])*rates[1]*y[5] - rates[4]*y[8], # HR, y[8]
                probabilities[3]*rates[3]*y[7] - rates[5]*y[9], # CD, y[9]
                (1-probabilities[3])*rates[3]*y[7] - rates[6]*y[10], # CR, y[10]
                rates[6]*y[10] - rates[7]*y[11], # M, y[11]
                probabilities[0] * y[0] * transmission_rates[0] * ( transmission_rates[1]*(y[1]+y[2]+y[3]+y[12]+y[13]+y[14]+y[15]) + (y[4]+y[5]+y[6]) ) / y[19] - rates[0]*y[12], # EA, y[12]
                rates[0]*y[12] - rates[0]*y[13],
                rates[0]*y[13] - rates[0]*y[14],
                rates[0]*y[14] - rates[8]*y[15], # IA, y[15]
                rates[2]*y[6] + rates[4]*y[8] + rates[7]*y[11] + rates[8]*y[15], # R, y[16]
                rates[5]*y[9] + rates[4]*y[20] - rates[9]*y[17], # D, y[17]
                rates[9]*y[17], # X, y[18]
                -rates[5]*y[9] - rates[4]*y[20], # N, y[19]
                probabilities[4]*rates[1]*y[5] - rates[4]*y[20]]) # HD, y[20]

        return dydt

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
            return np.inf
        else:
            transmission_rates = np.copy(self.transmission_rates)
            beta = transmission_rates[0]
            reduced_beta = position[0:3]*beta
            log_initial_infectious = position[3]
            # import pdb; pdb.set_trace()
            sigma_0 = position[4]
            step_size = 1.0
            N0 = self.initial_state[-2]
            pA = self.probabilities[0]

            Yt = np.zeros((len(self.control_dates)+1,self.number_of_states))
            # import pdb; pdb.set_trace()
            Y0 = np.array([N0-np.exp(log_initial_infectious),(1-pA)*np.exp(log_initial_infectious),
                           0,0,0,0,0,0,0,0,0,0,pA*np.exp(log_initial_infectious),0,0,0,0,0,0,N0,0])
            Yt[0] = Y0
            # import pdb; pdb.set_trace()
            sol = solve_ivp(self.delayed_ode,
                            [0,self.control_dates[0]],
                            Y0,
                            t_eval=np.arange(0,self.control_dates[0]+step_size,step_size),
                            args=(self.rates,self.probabilities,transmission_rates))
            Ttemp = sol.t
            Ytemp = sol.y
            Yt[1] = np.array([item[-1] for item in Ytemp])
            # print(Yt)
            # import pdb; pdb.set_trace()
            Tall = Ttemp
            Yall = Ytemp
            # print(Tall)
            for ic in range(len(self.control_dates)-1):
                time_range = np.array([self.control_dates[ic],self.control_dates[ic+1]])
                transmission_rates[0] = reduced_beta[ic]
                # import pdb; pdb.set_trace()
                sol = solve_ivp(self.delayed_ode,
                                time_range,
                                Yt[ic+1],
                                t_eval=np.linspace(self.control_dates[ic],self.control_dates[ic+1],int((self.control_dates[ic+1]-self.control_dates[ic])/step_size)+1),
                                args=(self.rates,self.probabilities,transmission_rates))
                Ttemp = sol.t
                Ytemp = sol.y
                Yt[ic+2] = np.array([item[-1] for item in Ytemp])
                Tall = np.append(Tall,Ttemp[1:])
                Yall = np.append(Yall,np.array([item[1:] for item in Ytemp]),axis=-1)

            ihdatafit = np.arange(int(37/step_size),int(78/step_size),int(1/step_size))
            # import pdb; pdb.set_trace()
            log_likelihood = np.sum(st.nbinom.logpmf(self.data,self.rates[1]*Yall[5,ihdatafit]/(sigma_0-1),1/sigma_0))
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
