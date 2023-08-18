Network based statistics (NBS)
------------------------------

NBS `(Zalesky et al.,
2010) <https://www.sciencedirect.com/science/article/abs/pii/S1053811910008852?via%3Dihub>`__
is a nonparametric statistical test used to isolate the components of an
N x N undirected connectivity matrix that differ significantly between
two distinct populations. Each element of the connectivity matrix stores
a connectivity value and each member of the two populations possesses a
distinct connectivity matrix. A component of a connectivity matrix is
defined as a set of interconnected edges. The NBS comprises fours steps:

1. Perform a two-sample T-test at each edge indepedently to test the
   hypothesis that the value of connectivity between the two populations
   come from distributions with equal means.
2. Threshold the T-statistic available at each edge to form a set of
   suprathreshold edges.
3. Identify any components in the adjacency matrix defined by the set of
   suprathreshold edges. These are referred to as observed components.
   Compute the size of each observed component identified; that is, the
   number of edges it comprises.
4. Repeat K times steps 1-3, each time randomly permuting members of the
   two populations and storing the size of the largest component
   identified for each permuation. This yields an empirical estimate of
   the null distribution of maximal component size. A corrected p-value
   for each observed component is then calculated using this null
   distribution.

Further analysis could implement `NBS
predict <https://www.sciencedirect.com/science/article/pii/S1053811921008983>`__

Network control theory
----------------------

Graph theory metrics
--------------------

Network based atrophy spreading
-------------------------------
