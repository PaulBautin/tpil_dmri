# Microscale: Regional properties
* receptor similarity [ref](https://www.nature.com/articles/s41593-022-01186-3)
* gene coexpression[ref](https://www.nature.com/articles/s41593-022-01186-3#ref-CR15)
* microstructural similarity[ref](https://www.nature.com/articles/s41593-022-01186-3#ref-CR17)

# Macroscale: Brain connectivity
## Structural connectivity
## Functional connectivity
* temporal profile similarity[ref](https://www.nature.com/articles/s41593-022-01186-3#ref-CR16)
## Structural covariance
* anatomical covariance[ref](https://www.nature.com/articles/s41593-022-01186-3#ref-CR13)
* morphometric similarity[ref](https://www.nature.com/articles/s41593-022-01186-3#ref-CR14)

# Pipeline
scil_decompose_connectivity.py

# Method
## SIFT2 
The optimal weighted set of streamlines is determined so that the resulting weighted local orientation density of streamlines is as close as possible to the fibre ODFs estimated using spherical. 

## COMMIT2
Solves a global inverse problem to estimate a weight for each streamline by assuming a generative multi-compartment microstructure model for the measured data, which usually includes the intra- and extra-axonal spaces and free water. The global optimization problem is solved by using constant microstructure properties throughout each streamline trajectory

# SC difference cause
## Inflammation
[Inflammation-Related Functional and Structural Dysconnectivity as a Pathway to Psychopathology](https://www.biologicalpsychiatryjournal.com/article/S0006-3223(22)01715-2/fulltext)

## Brain disorder impact
[The connectomics of brain disorders](https://www.nature.com/articles/nrn3901)
