Local diffusion MRI metrics were computed for every subject on DWI
images using 4 different local reconstruction models: ## Diffusion
Tensor Imaging (DTI) DTI metrics computations are implemented directly
in *TractoFlow* using ``dipy.reconst.dti``
(`tutorial <https://dipy.org/documentation/1.5.0/examples_built/reconst_dti/#example-reconst-dti>`__)
and are based on the reconstruction of the diffusion signal with a
tensor model `quality control metrics: (Tournier,
2011) <https://doi.org/10.1002/mrm.22924>`__. In our case, using a
b-value shell 1000 mm2/s the model estimates diffusion anisotropy
metrics using a weighted least squares single-tensor fit.

FreeWater
=========

Freewater and freewater corrected DTI metrics are computed in the
*freewater_flow* pipeline and are based on the separation of the
diffusion signal contributions from freewater and the rest with a
bi-tensor model `(Pasternak,
2009) <https://doi.org/10.1002/mrm.22055>`__. In our case, using b-value
shells 300 and 1000 mm2/s the model estimates freewater water fraction
using Accelerated Microstructure Imaging via Convex Optimization
`(AMICO) <https://github.com/daducci/AMICO>`__.

Neurite Orientation Distribution Density Imaging (NODDI)
========================================================

NODDI metrics are computed with the *noddi_flow* pipeline and are based
on the reconstruction of the WM microstructure with a 3 compartment
model: intra-cellular, extra-cellular, and CSF compartments `(Zhang,
2012) <https://doi.org/10.1016/j.neuroimage.2012.03.072>`__. In our
case, using b-value shells 300, 1000 and 2000 mm2/s the model estimates
an orientation dispersion index also using Accelerated Microstructure
Imaging via Convex Optimization
`(AMICO) <https://github.com/daducci/AMICO>`__.

Fiber Orientation Distribution Function (fODF)
==============================================

fODF metrics computations are implemented directly in *TractoFlow* using
``dipy.reconst.csdeconv``
(`tutorial <https://dipy.org/documentation/1.5.0/examples_built/reconst_csd/#example-reconst-csd>`__)
and are based on the reconstruction of the fODF with constrained
spherical deconvolution `(Raffelt,
2012) <https://doi.org/10.1016/j.neuroimage.2011.10.045>`__. In our
case, using b-value shells 1000 and 2000 mm2/s the model estimates the
fODFs by applying a constrained spherical deconvolution on ODFs using a
single by default (manually entered) fiber response function.
