Data Analysis
====================

.. tip::
    Statistics should be presented: statistic (degrees of freedom) = value;
    P = value; effect size statistic = value; and per cent confidence intervals = values



General linear model
--------------------

t-tests, correlations, partial correlations, ANOVAs, MANOVAs,… are just
specific instances of the linear model.

Statistical parametric mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Parametric
statistical <https://en.wikipedia.org/wiki/Parametric_statistics>`__
models are assumed at each voxel, using the general linear model to
describe the data variability in terms of experimental and confounding
effects, with residual variability. Hypotheses expressed in terms of the
model parameters are assessed at each voxel with `univariate
statistics <https://en.wikipedia.org/wiki/Univariate_(statistics)>`__.

Dimensionality reduction
------------------------

Principal component analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Is commonly used for dimension reduction. An usage tutorial can be found
on `scikit
website <https://scikit-learn.org/stable/modules/decomposition.html#pca>`__.
In this study we developed multiple PCA metrics:

Associative Techniques
----------------------

Partial least squares (PLS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Statistical significance of the LVs is determined via permutation
testing. **permutation sample**: In a permutation test, a new data set,
called a permutation sample, is obtained by randomly reordering the rows
(i.e., observations) of X and leaving Y unchanged. The PLSC model used
to compute the fixed effect model is then recomputed for the permutation
sample to obtain a new matrix of singular values. This procedure is
repeated for a large number of permutation samples, say 1000 or 10,000.
The set of all the singular values provides a sampling distribution of
the singular values under the null hypothesis and, therefore can be used
as a null hypothesis test.

Bootstrap resampling is used to examine the contribution and reliability
of the input features to each LV.

Split-half resampling can optionally be used to assess the reliability
of the LVs.

A cross-validated framework can optionally be used to examine how
accurate the decomposition is when employed in a predictive framework.


Multi-modal approaches
----------------------

spatial correlation
~~~~~~~~~~~~~~~~~~~

molecular-enriched network
~~~~~~~~~~~~~~~~~~~~~~~~~~

in-silico whole brain modelling analyses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
