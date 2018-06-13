Installation
============

.. |GPflow| raw:: html

    <a href="https://github.com/GPflow/GPflow" target="_blank">GPflow</a>

.. |TensorFlow| raw:: html

    <a href="https://www.tensorflow.org/install/" target="_blank">TensorFlow</a>

**GrandPrix** has been implemented in the |GPflow| package which uses |TensorFlow|. Thus **GrandPrix** installation reqruires installation of |TensorFlow| and |GPflow| first.

* Install |TensorFlow|

::

    pip install tensorflow

* Install |GPflow|

::

    git clone https://github.com/GPflow/GPflow.git
    cd GPflow
    python setup.py install
    cd

* Install GrandPrix package

::

    git clone https://github.com/ManchesterBioinference/GrandPrix
    cd GrandPrix
    python setup.py install
    cd

Installing with a new environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have problems with installation use the following script for a detailed setup guide from a new python environment.

* Create a new environment

::

    conda create -n newEnv python=3.5

* Activate the new environment

::

    source activate newEnv

* Create a new directory

::

    mkdir newInstall
    cd newInstall

* Follow the regular installation process described above

Trouble shooting
~~~~~~~~~~~~~~~~
If you do not have sudo rights and get a ``Permission denied`` error, use the follwing scripts to install |TensorFlow|, |GPflow| and **GrandPrix**:

::

    pip install --user tensorflow
    python setup.py --user install
