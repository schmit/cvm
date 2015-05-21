#!/usr/bin/env python

from setuptools import setup, find_packages
import cvm

setup(
    name = 'cvm',
    version = str(cvm.__version__),
    description = 'PySpark module for Cascading Kernel SVM',
    author = 'Carlos Riquelme, Lan Huong, Sven Schmit',
    author_email = 'schmit@stanford.edu',
    url = 'https://github.com/schmit/cvm',
    packages = ['cvm'],
    scripts = ['bin/mnist.py'],
    install_requires=open('requirements.txt').read().split()
)
