# GrandPrix
GrandPrix is a python package implementing the approach described in the paper "GrandPrix: Scaling up the Bayesian GPLVM for single-cell data" by Sumon Ahmed, Magnus Rattray and Alexis Boukouvalas.

We have included a series of notebook reproducing most of the results in the paper. Only [GPflow](https://github.com/GPflow/GPflow) version 0.3.8 is supported on GrandPrix [`betaVersion`](https://github.com/ManchesterBioinference/GrandPrix/tree/master). The [`master`](https://github.com/ManchesterBioinference/GrandPrix/tree/master) branch works with the latest version of `GPflow`.

## Installation
<!--
1. Install tensorflow - 'pip install tensorflow'
1. Install GPflow. Only GPflow version 0.3.8 is support on GrandPrix beta version. See [here](https://github.com/GPflow/GPflow) for more ifnormation. 
-->
<!--
```
git clone https://github.com/GPflow/GPflow.git
cd GPflow
git checkout 0.3.9
pip install .
```
-->
```
mkdir ~/envs
virtualenv -p /usr/bin/python3.5 envs/oldgpflow
oldgpflow
alias oldgpflow="source ~/envs/oldgpflow/bin/activate"
mkdir oldgpflow
cd oldgpflow
pip install -Iv tensorflow==1.8.0
git clone https://github.com/GPflow/GPflow.git 
cd GPflow
git reset --hard 3065dee
pip install .
pip install ipython
pip install jupyter
pip install matplotlib
```
## Clone the betaVersion branch 

To download the betaVersion branch only, run the following:

```
git clone -b betaVersion https://github.com/ManchesterBioinference/GrandPrix.git
```


