import sys
# from . import GrandPrix

def fit_model(
        data,
        n_latent_dims=1,
        n_inducing_points=10,
        kernel={'name':'RBF', 'ls':1.0, 'var':1.0},
        mData=None,
        latent_prior_mean=None,
        latent_prior_var=1.,
        latent_mean=None,
        latent_var=0.1,
        inducing_inputs=None,
        fix_parameters = None,
        predict=None,
        jitter=1e-6,
        dtype='float64',
        **kwargs):

    sys.argv.append(dtype)
    # print(sys.argv)
    # print(len(sys.argv))

    from . import GrandPrixModel
    m = GrandPrixModel.GrandPrixModel(data, n_latent_dims, n_inducing_points, kernel, mData,
                 latent_prior_mean, latent_prior_var, latent_mean, latent_var, inducing_inputs, dtype)
    # from GrandPrixModel import GrandPrixModel
    # m = GrandPrixModel(data, n_latent_dims, n_inducing_points, kernel, mData,
    #                                   latent_prior_mean, latent_prior_var, latent_mean, latent_var, inducing_inputs,
    #                                   dtype)
    m.set_jitter_level(jitter)
    m.set_trainable(fix_parameters)
    m.build()

    maxitr = 1000
    disp = False
    if 'maxiter' in kwargs:
        maxitr = kwargs.pop('maxiter')
    if 'display' in kwargs:
        disp = kwargs.pop('display')

    m.fit(maxiter=maxitr, display=disp)
    posterior = m.get_latent_dims()

    if predict is not None:
        prediction = m.predict(predict)
        posterior = posterior + prediction
    sys.argv = sys.argv[:-1]
    del m
    return posterior

# import pandas as pd
# import numpy as np
# np.random.seed(10)
# Y = pd.read_csv('../data/Windram/WindramTrainingData.csv', index_col=[0]).T.values
# # Y = Y.astype(np.float32)
# print(Y.dtype)
# p = fit_model(Y, kernel={'name':'Matern32', 'ls':1.0, 'var':1.0}, predict=100, display=True)
# print(p)