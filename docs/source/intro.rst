.. GrandPrix documentation master file, created by
   sphinx-quickstart on Sun May 13 18:19:19 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GrandPrix
=============================================================

GrandPrix is a package for non-linear probabilistic dimension reduction in python, based on the `Gaussian Process Latent Variable Model (GPLVM) <http://jmlr.csail.mit.edu/papers/volume6/lawrence05a/lawrence05a.pdf>`_. GrandPrix uses sparse variational approximation and relies on a small number of parameters termed inducing or auxiliary variables to project data to lower dimensional spaces. Among a number of sparse approximation algorithms, GrandPrix has been designed by using the Variational Free Energy (VFE) approximation that tries to maximize a lower bound to the exact marginal likelihood in order to select the inducing points and model hyperparameters jointly. This approach minimizes the KL divergence between the variational GP and the full posterior GP which allows it to avoid overfitting as well as to approximate the exact GP posterior.

    -- Allows model fitting and prediction with informative prior over the latent space.

.. toctree::
   :maxdepth: 1
   :hidden:

   self

   Installation
   TutorialsAndExamples
   APIandArchitecture
