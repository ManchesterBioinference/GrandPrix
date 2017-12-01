import gpflow
import numpy as np
import  time
import sys, os

class GrandPrix(object):
    def __init__(self, Y, mData=None):
        self.Y = Y
        self.mData = mData
        self.N, self.D = Y.shape
        self.Q = 1
        self.M = 10

    def MapTo01(self, y):
        return (y.copy() - y.min(0)) / (y.max(0) - y.min(0))

    def initialize_priors(self, priors):
        if type(priors['Priormean']) is str:
            X_prior_mean = self.mData[priors['Priormean']].values[:, None]
        else:
            X_prior_mean = priors['Priormean']
        X_prior_var = priors['Priorvar'] * np.ones((self.N, self.Q))
        return (X_prior_mean, X_prior_var)

    def initialize_latent_dims(self, Xmean=None, Xvar=0.1):
        if Xmean is not None:
            if type(Xmean) is str:
                X_mean = self.mData[Xmean].values[:, None]
            else:
                X_mean = Xmean
        else:
            X_mean = self.MapTo01(gpflow.gplvm.PCA_reduce(self.Y, self.Q))
        X_var = Xvar * np.ones((self.N, self.Q))
        return (X_mean, X_var)

    # def initialize_inducing_points(self, Z=None):
    #     if Z is not None:
    #         Z = Z
    #     else:
    #         Z = np.random.permutation(self.X_mean.copy())[:self.M]
    #     return Z

    def initialize_variational_parameters(self, vParams):
        # print(vParams)
        flag = True
        # np.random.seed(10)
        while flag:
            try:
                if 'Xmean' in vParams:
                    if 'Xvar' in vParams:
                        X_mean, X_var = self.initialize_latent_dims(Xmean=vParams['Xmean'], Xvar=vParams['Xvar'])
                    else:
                        X_mean, X_var = self.initialize_latent_dims(Xmean=vParams['Xmean'])
                        # print(X_mean, X_var)
                elif 'Xvar' in vParams:
                    X_mean, X_var = self.initialize_latent_dims(Xvar=vParams['Xvar'])
                else:
                    X_mean, X_var = self.initialize_latent_dims()

                Z = vParams.setdefault('Z', np.random.permutation(X_mean.copy())[:self.M])
                flag = False
            except:
                vParams = {}
        # print(X_mean, X_var, Z)
        return (X_mean, X_var, Z)

    def initialize_kernel(self, kernel=None):
        kernelName = 'RBF'
        input_dim = self.Q
        ls = 1.
        var = 1.
        if kernel is not None:
            if 'name' in kernel:
                kernelName = kernel['name']
            if 'ls' in kernel:
                ls = kernel['ls']
            if 'var' in kernel:
                var = kernel['var']

        if kernelName == 'RBF':
            k = gpflow.ekernels.RBF(input_dim, lengthscales=ls, variance=var, ARD=True)
        elif kernelName == 'Matern32':
            k = gpflow.kernels.Matern32(input_dim, lengthscales=ls, variance=var)
            # k =  k + gpflow.kernels.White(input_dim, variance=0.01)
        elif kernelName == 'Periodic':
            k = gpflow.kernels.PeriodicKernel(input_dim)
            k.lengthscales = ls
            k.period = 1.
        else:
            k = gpflow.ekernels.RBF(input_dim)
        return  k

    def fixed_params(self, param_list):
        if 'll_var' in param_list:  self.m.likelihood.variance.fixed = True
        if 'Z' in param_list:  self.m.Z.fixed = True
        if 'X_mean' in param_list:  self.m.X_mean.fixed = True
        if 'X_var' in param_list:  self.m.X_var.fixed = True
        if 'k_ls' in param_list:  self.m.kern.lengthscales.fixed = True
        if 'k_var' in param_list:  self.m.kern.variance.fixed = True

    def build_model(self, priors=None, vParams=None, **kwargs):
        if 'latent_dims' in kwargs:
            self.Q = kwargs.pop('latent_dims')
        if 'n_inducing_points' in kwargs:
            self.M = kwargs.pop('n_inducing_points')

        if 'kernel' in kwargs:
            kern = self.initialize_kernel(kwargs.pop('kernel'))
        else:
            kern = self.initialize_kernel()

        X_mean, X_var, Z = self.initialize_variational_parameters(vParams)

        if priors is not None:
            X_prior_mean, X_prior_var = self.initialize_priors(priors)
            self.m = gpflow.gplvm.BayesianGPLVM(Y=self.Y, kern=kern, X_prior_mean=X_prior_mean, X_prior_var=X_prior_var,
                                   X_mean=X_mean.copy(), X_var=X_var, Z=Z.copy(), M=self.M)
        else:
            self.m = gpflow.gplvm.BayesianGPLVM(Y=self.Y, kern=kern, X_mean=X_mean.copy(), X_var=X_var, Z=Z.copy(), M=self.M)
        self.m.likelihood.variance = 0.01
        self.m.likelihood.variance.fixed = False

        if 'fixed' in kwargs:
            self.fixed_params(kwargs.pop('fixed'))

    def fit_model(self, maxiter=1000, display=False):
        import warnings
        warnings.filterwarnings('ignore')
        sys.stdout = open(os.devnull, 'w')
        t0 = time.time()
        _ = self.m.optimize(maxiter=maxiter, disp=display)
        self.fitting_time = time.time() - t0
        sys.stdout = sys.__stdout__

    def predict_posterior(self, Xnew):
        return self.m.predict_y(Xnew)

    def get_model_fitting_time(self):
        return self.fitting_time

    def get_latent_dims(self, latent_dims=1):
        return self.m.X_mean.value[:, 0:latent_dims]

    def get_pseudotime(self):
        return self.get_latent_dims()
