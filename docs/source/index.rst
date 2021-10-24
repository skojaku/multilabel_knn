.. multilabel_knn documentation master file, created by
   sphinx-quickstart on Tue Jan 26 16:30:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

multilabel_knn
==============

Multilabel classification task aims to predict 'tags' assigned to a given sample, where one sample can have multiple tags.
The k-nearest neighbor algorithms predict the tags by finding examples similar to the given sample, and then 
predict the frequent tags for the examples. Several variants exist depending on in what sense similar to the given sample, and 
how to threshold frequency of tags.

This python package covers some basic algorithms as well as some enhanced algorithms for multilabel-classification based on k-nearest neighbors. If you have issues and feature requests, please raise them through `Github <https://github.com/skojaku/multilabel_knn>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Install
*******
See `project page <https://github.com/skojaku/multilabel_knn>`_


Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
