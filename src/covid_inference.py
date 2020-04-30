import numpy as np
import scipy.stats as st
from scipy.special import factorial
import sys

def random_walk(model,number_of_samples,initial_position,step_size,proposal_covariance=np.eye(1),thinning_rate=1):
    """
    Random walk Metropolis Hastings which takes as input a model and returns a N x q matrix of MCMC samples, where N is the number of
    samples and q is the number of parameters. Proposals, x', are drawn centered from the current position, x, by
    x + h*sqrt(proposal_covariance)*normal(0,1), where h is the step_size

    Parameters
    ----------

    model : class
        the model which the user wants to perform inference on. It should contain the method 'log_likelihood'

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
    # make sure the model has necessary methods
    try:
        model.log_likelihood(initial_position)
    except AttributeError:
        print("Your model",type(model).__name__,"does not have the method 'log_likelihood'.",
                                                "Please define it to use a random walk.")
        sys.exit(1)

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
        # import pdb; pdb.set_trace()
        # progress measure
        if iteration_index%(number_of_iterations//10)==0:
            print("Progress: ",100*iteration_index//number_of_iterations,'%')

        if identity:
            proposal = current_position + step_size*np.random.normal(size=number_of_parameters)
        else:
            proposal = current_position + step_size*proposal_cholesky.dot(np.random.normal(size=number_of_parameters))

        proposal_log_likelihood = model.log_likelihood(proposal)

        # accept-reject step
        if(np.random.uniform() < np.exp(proposal_log_likelihood - current_log_likelihood)):
            current_position = proposal
            current_log_likelihood = proposal_log_likelihood
            accepted_moves += 1

        if iteration_index%thinning_rate == 0:
            mcmc_samples[np.int(iteration_index/thinning_rate)] = current_position

    print("Acceptance ratio:",accepted_moves/number_of_iterations)
    return mcmc_samples

def mala(model,number_of_samples,initial_position,step_size,proposal_covariance=np.eye(1),thinning_rate=1):
    """
    Metropolis adjusted Langevin algorithm which takes as input a model and returns a N x q matrix of MCMC samples, where N is the number of
    samples and q is the number of parameters. Proposals, x', are drawn centered from the current position, x, by
    x + h/2*proposal_covariance*log_likelihood_gradient + h*sqrt(proposal_covariance)*normal(0,1), where h is the step_size

    Parameters
    ----------

    model : class
        the model which the user wants to perform inference on. It should contain the method 'log_likelihood' and 'log_likelihood_gradient'

    number_of_samples : integer
        the number of samples the random walk proposes

    initial_position : numpy array
        starting value of the Markov chain

    proposal_covariance: numpy array
        a q x q matrix where q is the number of paramters in the model. For optimal sampling this
        should represent the covariance structure of the samples

    step size : double
        a tuning parameter in the proposal step. this is a user defined parameter, change in order to get acceptance ratio ~0.5

    thinning_rate : integer
        the number of samples out of which you will keep one. this parameter can be increased to reduce autocorrelation if required

    Returns
    -------

    mcmc_samples : numpy array
        an N x q matrix of MCMC samples, where N is the number of samples and q is the number of parameters. These
        are the accepted positions in parameter space

    """
    # make sure the model has necessary methods
    try:
        model.log_likelihood_gradient(initial_position)
    except AttributeError:
        print("Your model",type(model).__name__,"does not have the method 'log_likelihood_gradient'.",
                                                "Either define it, or use a random walk.")
        sys.exit(1)

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

    proposal_covariance_inverse = np.linalg.inv(proposal_covariance)

    # initialise samples matrix and acceptance ratio counter
    accepted_moves = 0
    mcmc_samples = np.zeros((number_of_samples,number_of_parameters))
    mcmc_samples[0] = initial_position
    number_of_iterations = number_of_samples*thinning_rate

    # initial markov chain
    current_position = initial_position
    current_log_likelihood = model.log_likelihood(current_position)
    current_log_likelihood_gradient = model.log_likelihood_gradient(current_position)

    for iteration_index in range(1,number_of_iterations):
        # progress measure
        if iteration_index%(number_of_iterations//10)==0:
            print("Progress: ",100*iteration_index//number_of_iterations,'%')

        if identity:
            proposal = current_position + step_size*current_log_likelihood_gradient/2 + np.sqrt(step_size)*np.random.normal(size=number_of_parameters)
        else:
            proposal = current_position + step_size*proposal_covariance.dot(current_log_likelihood_gradient)/2 + np.sqrt(step_size)*proposal_cholesky.dot(np.random.normal(size=number_of_parameters))

        # compute transition probabilities for acceptance step
        proposal_log_likelihood = model.log_likelihood(proposal)
        proposal_log_likelihood_gradient = model.log_likelihood_gradient(proposal)

        forward_helper_variable = proposal - current_position - step_size*proposal_covariance.dot(current_log_likelihood_gradient)/2
        backward_helper_variable = current_position - proposal - step_size*proposal_covariance.dot(proposal_log_likelihood_gradient)/2

        transition_kernel_pdf_forward = -np.transpose(forward_helper_variable).dot(proposal_covariance_inverse).dot(forward_helper_variable)/(2*step_size)
        transition_kernel_pdf_backward = -np.transpose(backward_helper_variable).dot(proposal_covariance_inverse).dot(backward_helper_variable)/(2*step_size)

        # accept-reject step
        if(np.random.uniform() < np.exp(proposal_log_likelihood - transition_kernel_pdf_forward - current_log_likelihood + transition_kernel_pdf_backward)):
            current_position = proposal
            current_log_likelihood = proposal_log_likelihood
            current_log_likelihood_gradient = proposal_log_likelihood_gradient
            accepted_moves += 1

        if iteration_index%thinning_rate == 0:
            mcmc_samples[np.int(iteration_index/thinning_rate)] = current_position

    print("Acceptance ratio:",accepted_moves/number_of_iterations)
    return mcmc_samples

def simplified_manifold_mala(model,number_of_samples,initial_position,step_size,thinning_rate=1,regularization_constant=1e+6):
    """
    Simplified Manifold Metropolis adjusted Langevin algorithm which takes as input a model and returns a N x q matrix
    of MCMC samples, where N is the number of samples and q is the number of parameters. Proposals, x', are drawn
    centered from the current position, x, by x + h/2*softabs_hessian_inverse*log_likelihood_gradient +
    h*sqrt(softabs_hessian_inverse)*normal(0,1), where h is the step_size (See Betancourt (2013))

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
        a tuning parameter in the proposal step. this is a user defined parameter, change in order to get acceptance ratio ~0.5

    thinning_rate : integer
        the number of samples out of which you will keep one. this parameter can be increased to reduce autocorrelation if required

    Returns
    -------

    mcmc_samples : numpy array
        an N x q matrix of MCMC samples, where N is the number of samples and q is the number of parameters. These
        are the accepted positions in parameter space

    """
    # make sure the model has necessary methods
    try:
        model.log_likelihood_gradient(initial_position)
    except AttributeError:
        print("Your model",type(model).__name__,"does not have the method 'log_likelihood_gradient'.",
                                                "Either define it, or use a random walk.")
        sys.exit(1)

    try:
        model.log_likelihood_hessian(initial_position)
    except AttributeError:
        print("Your model",type(model).__name__,"does not have the method 'log_likelihood_hessian'.",
                                                "Either define it, or use a random walk or mala.")
        sys.exit(1)

    # initialise the covariance proposal matrix
    number_of_parameters = len(initial_position)

    # initialise samples matrix and acceptance ratio counter
    accepted_moves = 0
    mcmc_samples = np.zeros((number_of_samples,number_of_parameters))
    mcmc_samples[0] = initial_position
    number_of_iterations = number_of_samples*thinning_rate

    # initialise markov chain
    current_position = initial_position
    current_log_likelihood = model.log_likelihood(current_position)
    current_log_likelihood_gradient = model.log_likelihood_gradient(current_position)
    # we use the negative hessian of the positive log target
    # and then regularize using the softabs metric, see Betancourt (2013)
    current_log_likelihood_hessian = -model.log_likelihood_hessian(current_position)
    current_hessian_eigvals, current_hessian_eigvectors = np.linalg.eig(current_log_likelihood_hessian)
    current_regularized_eigvals = current_hessian_eigvals*(1/np.tanh(regularization_constant*current_hessian_eigvals))
    current_sqrt_inverse_softabs_hessian = current_hessian_eigvectors.dot(np.diag(1/(np.sqrt(current_regularized_eigvals))))
    current_inverse_softabs_hessian = current_sqrt_inverse_softabs_hessian.dot(np.transpose(current_sqrt_inverse_softabs_hessian))
    current_softabs_hessian = current_hessian_eigvectors.dot(
                              np.diag(current_regularized_eigvals)).dot(
                              np.transpose(current_hessian_eigvectors))
    # current_hessian_cholesky = np.linalg.cholesky(current_softabs_hessian)

    for iteration_index in range(1,number_of_iterations):
        # progress measure
        if iteration_index%(number_of_iterations//10)==0:
            print("Progress: ",100*iteration_index//number_of_iterations,'%')

        proposal = current_position + (step_size*current_inverse_softabs_hessian.dot(current_log_likelihood_gradient)/2 +
                                           np.sqrt(step_size)*current_sqrt_inverse_softabs_hessian.dot(np.random.normal(size=number_of_parameters)))

        proposal_log_likelihood = model.log_likelihood(proposal)
        if proposal_log_likelihood == -np.inf:
            if iteration_index%thinning_rate == 0:
                mcmc_samples[np.int(iteration_index/thinning_rate)] = current_position
            continue

        proposal_log_likelihood_gradient = model.log_likelihood_gradient(proposal)
        # we use the negative hessian of the positive log target
        # and then regularize using the softabs metric, see Betancourt (2013)
        proposal_log_likelihood_hessian = -model.log_likelihood_hessian(proposal)

        proposal_hessian_eigvals, proposal_hessian_eigvectors = np.linalg.eig(proposal_log_likelihood_hessian)
        proposal_regularized_eigvals = proposal_hessian_eigvals*(1/np.tanh(regularization_constant*proposal_hessian_eigvals))
        proposal_sqrt_inverse_softabs_hessian = proposal_hessian_eigvectors.dot(np.diag(1/(np.sqrt(proposal_regularized_eigvals))))
        proposal_inverse_softabs_hessian = proposal_sqrt_inverse_softabs_hessian.dot(np.transpose(proposal_sqrt_inverse_softabs_hessian))
        proposal_softabs_hessian = proposal_hessian_eigvectors.dot(
                                  np.diag(proposal_regularized_eigvals)).dot(
                                  np.transpose(proposal_hessian_eigvectors))

        forward_helper_variable = proposal - current_position - step_size*current_inverse_softabs_hessian.dot(current_log_likelihood_gradient)/2
        backward_helper_variable = current_position - proposal - step_size*proposal_inverse_softabs_hessian.dot(proposal_log_likelihood_gradient)/2

        transition_kernel_pdf_forward = 0.5*np.sum(np.log(current_regularized_eigvals))-np.transpose(forward_helper_variable).dot(current_softabs_hessian).dot(forward_helper_variable)/(2*step_size)
        transition_kernel_pdf_backward = 0.5*np.sum(np.log(proposal_regularized_eigvals))-np.transpose(backward_helper_variable).dot(proposal_softabs_hessian).dot(backward_helper_variable)/(2*step_size)

        if(np.random.uniform() < np.exp(proposal_log_likelihood - transition_kernel_pdf_forward - current_log_likelihood + transition_kernel_pdf_backward)):
            current_position = proposal
            current_log_likelihood = proposal_log_likelihood
            current_log_likelihood_gradient = proposal_log_likelihood_gradient
            current_regularized_eigvals = proposal_regularized_eigvals
            current_sqrt_inverse_softabs_hessian = proposal_sqrt_inverse_softabs_hessian
            current_inverse_softabs_hessian = proposal_inverse_softabs_hessian
            current_softabs_hessian = proposal_softabs_hessian
            accepted_moves += 1

        if iteration_index%thinning_rate == 0:
            mcmc_samples[np.int(iteration_index/thinning_rate)] = current_position

    print("Acceptance ratio:",accepted_moves/number_of_iterations)
    return mcmc_samples
