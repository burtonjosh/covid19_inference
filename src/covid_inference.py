import numpy as np
from numba import jit

class normal:
    def __init__(self,mean,variance):
        self.mean = mean
        self.variance = variance

    def log_likelihood(self, position):
        return -np.power((position-self.mean),2)/(2*self.variance)

    def log_likelihood_gradient(self,position):
        return -(position-self.mean)/(self.variance)

    def log_likelihood_hessian(self,position):
        return -1/self.variance

def random_walk(model,number_of_samples,initial_position,step_size,proposal_covariance=None,thinning_rate=1):
    """
    Random walk Metropolis Hastings which takes as input a model and returns a N x q matrix of MCMC samples, where N is the number of
    samples and q is the number of parameters. Proposals, x', are drawn centered from the current position, x, by
    x + h*sqrt(proposal_covariance)*normal(0,1), where h is the step_size

    Parameters
    ----------

    model : class.
        the model which the user wants to perform inference on. It should contain the method log_target

    number_of_samples : integer.
        the number of samples the random walk proposes

    initial_position : numpy array.
        starting value of the Markov chain

    proposal_covariance: numpy array.
        a q x q matrix where q is the number of paramters in the model. For optimal sampling this
        should represent the covariance structure of the samples

    step size : double.
        a tuning parameter in the proposal step. this is a user defined parameter, change in order to get acceptance ratio ~0.234

    thinning_rate : integer.
        the number of samples out of which you will keep one. this parameter can be increased to reduce autocorrelation if required

    Returns
    -------

    mcmc_samples : numpy array.
        an N x q matrix of MCMC samples, where N is the number of samples and q is the number of parameters. These
        are the accepted positions in parameter space

    """
    number_of_parameters = len(initial_position)
    if proposal_covariance == None:
        proposal_covariance = np.eye(number_of_parameters)

    if np.equal(proposal_covariance,np.eye(number_of_parameters)):
        identity = True
    else:
        identity = False
        proposal_cholesky = np.linalg.cholesky(proposal_covariance)

    accepted_moves = 0
    mcmc_samples = np.zeros((number_of_samples,number_of_parameters))

    # initialise samples matrix
    mcmc_samples[0] = initial_position
    number_of_iterations = number_of_samples*thinning_rate

    current_position = initial_position
    current_log_likelihood = model.log_likelihood(current_position)

    for iteration in range(1,number_of_iterations):
        # progress measure
        if iteration%(number_of_iterations//10)==0:
            print("Progress: ",100*iteration/number_of_iterations)

        if identity:
            proposal = current_position + step_size*np.random.normal(size=number_of_parameters)
        else:
            proposal = current_position + step_size*proposal_cholesky.dot(np.random.normal(size=number_of_parameters))

        proposal_log_likelihood = model.log_likelihood(proposal)

        if(np.random.uniform() < np.exp(proposal_log_likelihood - current_log_likelihood)):
            current_position = proposal
            current_log_likelihood = proposal_log_likelihood
            accepted_moves += 1

        if iteration%thinning_rate == 0:
            mcmc_samples[np.int(iteration/thinning_rate)] = current_position

    print(accepted_moves/number_of_iterations)

    return mcmc_samples
