import numpy as np
import time
import sys, os
from shutil import copyfile

# print('I am in the model...')
# print(sys.argv)
# print(sys.argv[-1])
# l = len(sys.argv)
# if sys.argv[-1] == 'float32':
#     copyfile('gpflowrc32', 'gpflowrc')
# else:
#     copyfile('gpflowrc64', 'gpflowrc')
import tensorflow as tf
#
import gpflow

def MapTo01(y):
    return (y.copy() - y.min(0)) / (y.max(0) - y.min(0))

class GrandPrixModel(object):
    """
    GrandPrix: Scaling up the Bayesian Gaussian Process Latent Variable Model (BGPLVM).
    
    Non-linear probabilistic dimension reduction that uses sparse variational approximation 
    to project data to a lower dimensional space.
    
    Sparse approximations are useful techniques for practical inference of Gaussian processes (GP) to deal 
    with large datasets. Sparse approximations rely on a small number of parameters termed inducing 
    or auxiliary points that approximate the posterior distribution over functions. GrandPrix uses 
    the Variational Free Energy (VFE) approximation that tries to maximize a lower bound to the 
    exact marginal likelihood in order to select the inducing points and model hyperparameters jointly.
    This approach minimizes the KL divergence between the variational GP and the full posterior GP which allows it
    to avoid overfitting as well as to approximate the exact GP posterior.
    
    -- Allows model fitting and prediction with informative prior over the latent space.
    
    Parameters
    ----------
    data: array-like, shape N x D
        Observed data, where N is the number of samples and D is the number of features.
        
    n_latent_dims: int, optional (default: 1)
        Number of latent dimentions to compute.
        
    n_inducing_points: int, optional (default: 10)
        Number of inducing or auxiliary points. 
          
    kernel: gpflow.kernels object, optional (default: RBF kernel with lengthscale and variance set to 1.0)
        Kernel functions are used to compute the covariance among datapoints. They impose constraints such as smoothness, periodicity on the function being
        learned that is shared by all datapoints.
        Kernels are parameterize by a set of hyperparameters, i.e. lengthscale, variance, etc which can be optimized during model fitting.
    
    latent_prior_mean: array-like, shape N x n_latent_dims, optional (default: 0)
    
    latent_prior_var: array-like, shape N x n_latent_dims, optional (default: 1.)
    
    latent_mean: array-like, shape N x n_latent_dims, optional (default: PCA)
        Initial mean values of the distribution over the latent dimensions.
        
    latent_var: array-like, shape N x n_latent_dims, optional (default: 0.1)
        Initial variance of the distribution over the latent dimensions.
        
    inducing_inputs: array-like, shape n_inducing_points x n_latent_dims, optional (default: randome subset from laten_mean)
    
    References
    ----------
    
    """
    def __init__(self, data, n_latent_dims=1, n_inducing_points=10, kernel={'name':'RBF', 'ls':1.0, 'var':1.0}, mData=None,
                 latent_prior_mean=None, latent_prior_var=1., latent_mean=None, latent_var=0.1, inducing_inputs=None, dtype='float64'):
        self.Y = None
        self.Q = n_latent_dims
        self.M = n_inducing_points
        self.kern = None
        self.mData = mData
        self.X_prior_mean = None
        self.X_prior_var = None
        self.X_mean = None
        self.X_var = None
        self.Z = None

        self.set_Y(data)
        self.N, self.D = self.Y.shape

        self.set_kern(kernel)

        self.set_X_prior_mean(latent_prior_mean)
        self.set_X_prior_var(latent_prior_var)

        self.set_X_mean(latent_mean)
        self.set_X_var(latent_var)

        self.set_inducing_inputs(inducing_inputs)

        self.fitting_time = 0

        with gpflow.defer_build():
            self.m = gpflow.models.BayesianGPLVM(Y=self.Y, kern=self.kern, X_prior_mean=self.X_prior_mean, X_prior_var=self.X_prior_var,
                                            X_mean=self.X_mean.copy(), X_var=self.X_var.copy(), Z=self.Z.copy(), M=self.M)
            self.m.likelihood.variance = 0.01

    def __str__(self):
        return "%s"%(self.m)

    def build(self):
        """
        Build the model into a Tensorflow graph.
        :return: 
        """
        self.m.compile()

    def fit(self, maxiter=1000, display=False):
        """
        Fit the BGPLVM model.
        :param maxiter: int, optional (default: 1000)
            Maximum number of iterations to perform.
        :param display: bool, optional (default: False)
            If set to True, print convergence messages. 
        """
        opt = gpflow.train.ScipyOptimizer()

        try:
            t0 = time.time()
            opt.minimize(self.m, maxiter=maxiter, disp=display)
            self.fitting_time = time.time() - t0
        except:
            print('Warning: The model terminates abnormally...')


    def predict(self, Xnew):
        """
        Predict posterior mean and variance using the BGPLVM. The prediction can also be done on the unfitted model using the Gaussian Process prior. 
        :param 
            Xnew: array-like, shape n_sample x n_latent_dims
                n_sample is the number of query points where the prediction will be evaluated.
        :return:
            data_mean: array-like, shape N x D
                Mean values of the predictive distribution at the query points.
            data_var: array-like, shape N x D
                Variance of the predictive distribution at the query points.
        """
        if type(Xnew) is int:
            latent_mean = self.get_latent_dims()[0]
            n_points = Xnew
            del Xnew
            Xnew = np.linspace(min(latent_mean), max(latent_mean), n_points)[:, None]
        # assert isinstance(Xnew, np.ndarray)
        return self.m.predict_y(Xnew)

    def get_latent_dims(self):
        """
        Get predictive distribuiton over latent dimensions of the Gaussian Process Latent Variable Model.
        :return: 
            latent_mean: array-like, shape N x n_latent_dims
                Mean values of the predictive distribution over the latent dimensions.
            latent_var: array-like, shape N x n_latent_dims
                Variance of the predictive distribution over the latent dimensions.
        """
        return (self.m.X_mean.read_value()[:, 0:self.Q], self.m.X_var.read_value()[:, 0:self.Q])

    def get_model(self):
        return self.m.as_pandas_table()

    def set_trainable(self, paramlist=None):
        if paramlist is not None:
            if 'kernel_lengthscales' in paramlist:  self.m.kern.lengthscales.trainable = False
            if 'kernel_variance' in paramlist:  self.m.kern.variance.trainable = False
            if 'likelihood_variance' in paramlist: self.m.likelihood.variance.trainable = False
            if 'inducing_inputs' in paramlist:  self.m.feature.Z.trainable = False
            if 'latent_mean' in paramlist:  self.m.X_mean.trainable = False
            if 'latent_variance' in paramlist: self.m.X_var.trainable = False

    def set_jitter_level(self, jitter_level):
        gpflow.settings.numerics.jitter_level = jitter_level

    def set_Y(self, data):
        self.Y = data

    def set_kern(self, kernel):
        if kernel is not None:
            if 'name' in kernel:
                kernelName = kernel['name']
            if 'ls' in kernel:
                ls = kernel['ls']
            if 'var' in kernel:
                var = kernel['var']

        if kernelName == 'RBF':
            k = gpflow.kernels.RBF(self.Q, lengthscales=ls, variance=var, ARD=True)
        elif kernelName == 'Matern32':
            k = gpflow.kernels.Matern32(self.Q, lengthscales=ls, variance=var)
            # k =  k + gpflow.kernels.White(input_dim, variance=0.01)
        elif kernelName == 'Periodic':
            k = gpflow.kernels.Periodic(self.Q)
            k.lengthscales = ls
            k.period = 1.

        self.kern = k

    def set_X_prior_mean(self, X_prior_mean):
        if X_prior_mean is not None:
            if type(X_prior_mean) is str:
                self.X_prior_mean = self.mData[X_prior_mean].values[:, None]
            else:
                self.X_prior_mean = X_prior_mean
        else:
            self.X_prior_mean = np.zeros((self.N, self.Q))

    def set_X_prior_var(self, X_prior_var):
        if X_prior_var is not None:
            if type(X_prior_var) is str:
                self.X_prior_var = self.mData[X_prior_var].values[:, None]
            elif type(X_prior_var) is np.ndarray:
                self.X_prior_var = X_prior_var
            else:
                self.X_prior_var = X_prior_var * np.ones((self.N, self.Q))
        else:
            self.X_prior_var = 1. * np.ones((self.N, self.Q))

    def set_X_mean(self, X_mean):
        if X_mean is not None:
            if type(X_mean) is str:
                self.X_mean = self.mData[X_mean].values[:, None]
            else:
                self.X_mean = X_mean
        else:
            self.X_mean = MapTo01(gpflow.models.PCA_reduce(self.Y, self.Q))

    def set_X_var(self, X_var):
        if type(X_var) is str:
            self.X_var = self.mData[X_var].values[:, None]
        elif type(X_var) is np.ndarray:
            self.X_var = X_var
        else:
            self.X_var = X_var * np.ones((self.N, self.Q))

    def set_inducing_inputs(self, Z):
        if Z is None:
            self.Z = np.random.permutation(self.X_mean.copy())[:self.M]
        else:
            self.Z = np.asarray(Z)