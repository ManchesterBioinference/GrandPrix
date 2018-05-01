 # GrandPrix

GrandPrix is a package for non-linear probabilistic dimension reduction algorithm in python, using [TensorFlow](github.com/tensorflow) and [GPFlow](https://github.com/GPflow/GPflow). GrandPrix uses sparse variational approximation to project data to lower dimensional spaces. The model is described in the paper

["GrandPrix: Scaling up the Bayesian GPLVM for single-cell data.", 
Sumon Ahmed, Magnus Rattray and Alexis Boukouvalas, bioRxiv, 2017.](https://www.biorxiv.org/content/early/2017/12/03/227843)

## Installation
If you have any problems with installation see the script at the bottom of the page for a detailed setup guide from a new python environment. 

   - Install tensorflow
```
pip install tensorflow
```
   - Install GPflow
```
git clone https://github.com/GPflow/GPflow.git
cd GPflow    
pip install .
cd
```
    
See [GPFlow](https://github.com/GPflow/GPflow) page for more detailed instructions.

   - Install GrandPrix package
```
git clone https://github.com/ManchesterBioinference/GrandPrix
cd GrandPrix
python setup.py install
cd
```
<!--
## Documentation
The online documentation for GrandPrix is available here:
-  [Online documentation](./docs/_build/html/index.html)
-->
## List of notebooks
To run the notebooks
```
cd GrandPrix/notebooks
jupyter notebook
```

| File <br> name | Description | 
| --- | --- | 
| <a href="./notebooks/Windram.ipynb" target="_blank">Windram</a>| Application of GrandPrix to microarray data. |
| [McDavid](./McDavid.ipynb)       | Application of GrandPrix to cell cycle data. |
| [Shalek](./notebooks/Shalek.ipynb)| Application of GrandPrix to single-cell RNA_seq from mouse dentritic cells. |
| [Droplet_DPT](./notebooks/Droplet_DPT.ipynb)| Application of GrandPrix to droplet based single-cell RNA_seq data. |
| [Guo](./notebooks/Guo.ipynb)| Application of extendend 2-D GrandPrix model to embryonic stem cells.|
<!--
| Zheng| Sampling from the BGP model. |
-->
## Running in a cluster
When running GrandPrix in a cluster it may be useful to constrain the number of cores used. To do this insert this code at the beginning of your script.
```
from gpflow import settings
settings.session.intra_op_parallelism_threads = NUMCORES
settings.session.inter_op_parallelism_threads = NUMCORES
```
## Installing with a new environment

-  Create a new environment
```
conda create -n newEnv python=3.5
```
-  Activate the new environment
```
source activate newEnv
```
-  Create a new directory
```
mkdir newInstall
cd newInstall
```
-  Follow the regular installation process described above
