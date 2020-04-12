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

        mean : numpy array
            a 1x5 mean vector of the hybrid rosenbrock distribution

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
        # H = function(q) {-matrix(c(-2*a -24*b*q[1]^2 +4*b*q[2] +4*b*q[4], 4*b*q[1], 0, 4*b*q[1], 0,
        #                            4*b*q[1], -2*b*(1 +6*q[2]^2 -2*q[3]), 4*b*q[2], 0, 0,
        #                            0, 4*b*q[2], -2*b, 0, 0,
        #                            4*b*q[1], 0, 0, -2*b*(1 +6*q[4]^2 -2*q[5]), 4*b*q[4],
        #                            0, 0, 0, 4*b*q[4], -2*b) ,d,d, byrow = T)}

        log_likelihood_hessian = np.array([[-2*self.shape[0] - 24*self.shape[1]*position[0]**2 + 4*self.shape[1]*position[1]+4*self.shape[1]*position[3],
                                            4*self.shape[1]*position[0],0,4*self.shape[1]*position[0],0],
                                           [4*self.shape[1]*position[0],-2*self.shape[1]*(1 + 6*position[1]**2 - 2*position[2]),4*self.shape[1]*position[1],0,0],
                                           [0,4*self.shape[1]*position[1],-2*self.shape[1],0,0],
                                           [4*self.shape[1]*position[0],0,0,-2*self.shape[1]*(1 + 6*position[3]**2 - 2*position[4]),4*self.shape[1]*position[3]],
                                           [0,0,0,4*self.shape[1]*position[3],-2*self.shape[1]]])
        return log_likelihood_hessian

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

def simple_manifold_mala(model,number_of_samples,initial_position,step_size,proposal_covariance=np.eye(1),thinning_rate=1):
    """
    Simple Manifold Metropolis adjusted Langevin algorithm which takes as input a model and returns a N x q matrix
    of MCMC samples, where N is the number of samples and q is the number of parameters. Proposals, x', are drawn
    centered from the current position, x, by x + h/2*proposal_covariance*log_likelihood_gradient +
    h*sqrt(proposal_covariance)*normal(0,1), where h is the step_size

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
    current_log_likelihood_hessian = -model.log_likelihood_hessian(current_position)
    # import pdb; pdb.set_trace()
    current_hessian_cholesky = np.linalg.cholesky(current_log_likelihood_hessian)

    for iteration_index in range(1,number_of_iterations):
        # progress measure
        if iteration_index%(number_of_iterations//10)==0:
            print("Progress: ",100*iteration_index//number_of_iterations,'%')

        if identity:
            proposal = current_position + (step_size*current_log_likelihood_hessian.dot(current_log_likelihood_gradient)/2 +
                                           np.sqrt(step_size)*current_hessian_cholesky.dot(np.random.normal(size=number_of_parameters)))
        else:
            proposal = current_position + (step_size*proposal_covariance.dot(current_log_likelihood_hessian).dot(current_log_likelihood_gradient)/2 +
                                           np.sqrt(step_size)*proposal_cholesky.dot(current_hessian_cholesky).dot(np.random.normal(size=number_of_parameters)))

        proposal_log_likelihood = model.log_likelihood(proposal)
        proposal_log_likelihood_gradient = model.log_likelihood_gradient(proposal)
        proposal_log_likelihood_hessian = -model.log_likelihood_hessian(proposal)
        # import pdb; pdb.set_trace()
        proposal_hessian_cholesky = np.linalg.cholesky(proposal_log_likelihood_hessian)

        forward_helper_variable = proposal - current_position - step_size*proposal_covariance.dot(current_log_likelihood_hessian).dot(current_log_likelihood_gradient)/2
        backward_helper_variable = current_position - proposal - step_size*proposal_covariance.dot(proposal_log_likelihood_hessian).dot(proposal_log_likelihood_gradient)/2

        transition_kernel_pdf_forward = -np.transpose(forward_helper_variable).dot(proposal_covariance_inverse).dot(forward_helper_variable)/(2*step_size)
        transition_kernel_pdf_backward = -np.transpose(backward_helper_variable).dot(proposal_covariance_inverse).dot(backward_helper_variable)/(2*step_size)

        if(np.random.uniform() < np.exp(proposal_log_likelihood - transition_kernel_pdf_forward - current_log_likelihood + transition_kernel_pdf_backward)):
            current_position = proposal
            current_log_likelihood = proposal_log_likelihood
            current_log_likelihood_gradient = proposal_log_likelihood_gradient
            current_log_likelihood_hessian = proposal_log_likelihood_hessian
            accepted_moves += 1

        if iteration_index%thinning_rate == 0:
            mcmc_samples[np.int(iteration_index/thinning_rate)] = current_position

    print("Acceptance ratio:",accepted_moves/number_of_iterations)
    return mcmc_samples
