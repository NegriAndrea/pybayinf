from setuptools import setup
from setuptools import find_packages
import os

# Optional project description in README.md:

current_directory = os.path.dirname(os.path.abspath(__file__))

try:

    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:

        long_description = f.read()

except Exception:

    long_description = 'EAGLE-like post-process code'

setup(

# Project name:
name='pybayinf',

# Packages to include in the distribution:
packages=find_packages(','),

# Project version number:
version='0.1',

# List a license for the project, eg. MIT License
license='',

# Short description of your library:
description='Bayesian luminosity function code',

# Long description of your library:

long_description=long_description,

long_description_content_type='text/markdown',

# Your name:
author='Andrea Negri',

# Your email address:
author_email='anegri@iac.es',

# Link to your github repository or website:
url='https://github.com/NegriAndrea',

# Download Link from where the project can be downloaded from:
download_url='https://github.com/NegriAndrea',


# List project dependencies:
install_requires=['numpy','astropy', 'scipy',
    'h5py>=3.2.0','emcee', 'pathlib', 'matplotlib', 'tqdm', 'corner'],

# https://pypi.org/classifiers/
# classifiers=[]

)
