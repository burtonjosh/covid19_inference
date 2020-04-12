import numpy as np

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

def random_walk(model,number_of_samples,initial_position,step_size,proposal_covariance=np.eye(1),thinning_rate=1):
    """
    Random walk Metropolis Hastings which takes as input a model and returns a N x q matrix of MCMC samples, where N is the number of
    samples and q is the number of parameters. Proposals, x', are drawn centered from the current position, x, by
    x + h*sqrt(proposal_covariance)*normal(0,1), where h is the step_size

    Parameters
    ----------

    model : class
        the model which the user wants to perform inference on. It should contain the method 'log_likelihood'.

    number_of_samples : integer
        the number of samples the random walk proposes

    initial_position : numpy array
        starting value of the Markov chain

    proposal_covariance: numpy array
        a q x q matrix where q is the number of paramters in the model. For optimal sampling this
        should represent the covariance structure of the samples

    step size : double
        a tuning parameter in the proposal step. this is a user defined parameter, change in order to get acceptance ratio ~0.234

    thinning_rate : integer
        the number of samples out of which you will keep one. this parameter can be increased to reduce autocorrelation if required

    Returns
    -------

    mcmc_samples : numpy array
        an N x q matrix of MCMC samples, where N is the number of samples and q is the number of parameters. These
        are the accepted positions in parameter space

    """
    # initialise the covariance proposal matrix
    number_of_parameters = len(initial_position)

    # check if default value is used, and set to q x q identity
    if np.array_equal(proposal_covariance, np.eye(1)):
        proposal_covariance = np.eye(number_of_parameters)

    if np.array_equal(proposal_covariance, np.eye(number_of_parameters)):
        identity = True
    else:
        identity = False
        proposal_cholesky = np.linalg.cholesky(proposal_covariance)

    # initialise samples matrix and acceptance ratio counter
    accepted_moves = 0
    mcmc_samples = np.zeros((number_of_samples,number_of_parameters))
    mcmc_samples[0] = initial_position
    number_of_iterations = number_of_samples*thinning_rate

    # initial markov chain
    current_position = initial_position
    current_log_likelihood = model.log_likelihood(current_position)

    for iteration_index in range(1,number_of_iterations):
        # progress measure
        if iteration_index%(number_of_iterations//10)==0:
            print("Progress: ",100*iteration_index//number_of_iterations,'%')

        if identity:
            proposal = current_position + step_size*np.random.normal(size=number_of_parameters)
        else:
            proposal = current_position + step_size*proposal_cholesky.dot(np.random.normal(size=number_of_parameters))

        proposal_log_likelihood = model.log_likelihood(proposal)

        if(np.random.uniform() < np.exp(proposal_log_likelihood - current_log_likelihood)):
            current_position = proposal
            current_log_likelihood = proposal_log_likelihood
            accepted_moves += 1

        if iteration_index%thinning_rate == 0:
            mcmc_samples[np.int(iteration_index/thinning_rate)] = current_position

    print("Acceptance ratio",accepted_moves/number_of_iterations)
    return mcmc_samples

def mala():
    return mcmc_samples

def single_manifold_mala():
    return mcmc_samples
