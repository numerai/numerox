#!/usr/bin/env python

import os
from setuptools import setup, find_packages


SHORT_DESCRIPTION = ("Numerai tournament toolbox written in Python")


def get_long_description():
    with open('readme.rst', 'r') as fid:
        long_description = fid.read()
    idx = max(0, long_description.find("Numerox is a Numerai"))
    long_description = long_description[idx:]
    return long_description


def get_version_str():
    ver_file = os.path.join('numerox', 'version.py')
    with open(ver_file, 'r') as fid:
        version = fid.read()
    version = version.split("= ")
    version = version[1].strip()
    version = version.strip("\"")
    return version


CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 "
    "or later (GPLv3+)",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering"]


REQUIRES = ['numpy',
            'scipy',
            'pandas',
            'tables',
            'scikit-learn',
            'numerapi',
            'nose']


metadata = dict(name='numerox',
                maintainer="Keith Goodman",
                description=SHORT_DESCRIPTION,
                long_description=get_long_description(),
                url="https://github.com/kwgoodman/numerox",
                license="GNU General Public License v3",
                classifiers=CLASSIFIERS,
                platforms="OS Independent",
                version=get_version_str(),
                packages=find_packages(),
                package_data={'numerox': ['LICENSE', 'readme.rst',
                                          'release.rst', 'tests/test_data.hdf',
                                          'tests/tiny_dataset_csv.zip']},
                install_requires=REQUIRES)
setup(**metadata)
