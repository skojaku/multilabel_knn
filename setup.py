#!/usr/bin/env python
# encoding: utf-8
import os
from os import path

from setuptools import find_packages, setup

__version__ = "0.0.5"


def load_requires_from_file(fname):
    if not os.path.exists(fname):
        raise IOError(fname)
    return [pkg.strip() for pkg in open(fname, "r")]


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setup(
    name="multilabel_knn",
    version=__version__,
    author="Sadamori Kojaku",
    url="https://github.com/skojaku/multilabel_knn",
    description="A lightweight toolbox for multilabel classification algorithms based on the k-nearest neighbors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    #install_requires=load_requires_from_file("requirements.txt"),
    include_package_data=True,
    zip_safe=False,
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="word embedding, graph embedding",
)
