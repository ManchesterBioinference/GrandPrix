from distutils.core import setup

DESCRIPTION = "GrandPrix: Scaling up the Bayesian GPLVM."
LONG_DESCRIPTION = "GrandPrix: Scaling up the Bayesian GPLVM for single-cell data."
NAME = "GrandPrix"
AUTHOR = "Sumon Ahmed"
AUTHOR_EMAIL = "sumon.ahmed@postgrad.manchester.ac.uk"
MAINTAINER = "Sumon Ahmed"
MAINTAINER_EMAIL = "sumon.ahmed@postgrad.manchester.ac.uk"
DOWNLOAD_URL = 'https://github.com/ManchesterBioinference/GrandPrix'
LICENSE = 'MIT'

VERSION = '0.1'

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      url=DOWNLOAD_URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      packages=['GrandPrix'],
      package_data={'':['gpflowrc', 'gpflowrc32', 'gpflowrc64']},
      include_package_data=True
      )
