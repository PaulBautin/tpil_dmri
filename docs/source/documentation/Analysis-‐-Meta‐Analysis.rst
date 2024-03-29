Meta Analysis
================================================

Best practices
----------------------------------------------------------------

1. Using a sufficient sample of included studies (n ≥ 17) in
   meta-analyses on neuroimaging is necessary to have the power to
   detect small effects and to avoid results being driven by a small
   number of studies.
2. The sample size of healthy controls and patient groups included in
   individual experiments must be sufficient (n ≥ 10 per group).
3. It is critical that studies included in a neuroimaging meta-analysis
   use a whole-brain analytical approach. The inclusion of experiments
   that only report region-of-interest (ROI) analyses may result in an
   over-representation of these regions.
4. Differing approaches to control for multiple comparisons in
   neuroimaging studies can introduce variability in the reported
   findings. In meta-analyses, this problem can be present at 2 levels:
   (i) The inclusion of experiments that lack adequate control of type I
   error will inflate apparent structural differences attributed to
   chronic pain. (ii) Meta-analyses themselves need to rigorously
   control for multiple comparisons across locations in the brain.
   Current standards for meta-analyses recommend cluster-wise
   family-wise error (cFWE) correction as the gold standard.

Neurosynth
----------------------------------------------------------------

-  **uniformity test map**: z-scores from a one-way ANOVA testing
   whether the proportion of studies that report activation at a given
   voxel differs from the rate that would be expected if activations
   were uniformly distributed throughout gray matter. The uniformity
   test map can be interpreted in roughly the same way as most standard
   whole-brain fMRI analysis: it displays the degree to which each voxel
   is consistently activated in studies that use a given term. For
   instance, the fact that the uniformity test map for the term
   '`emotion <https://neurosynth.org/analyses/terms/emotion>`__'
   displays high z-scores in the amygdala implies that studies that use
   the word emotion a lot tend to consistently report activation in the
   amygdala--at least, more consistently than one would expect if
   activation were uniformly distributed throughout gray matter. Note
   that, unlike most meta-analysis packages (e.g., ALE or MKDA),
   z-scores aren't generated through permutation, but using a chi-square
   test. (chi-square test is used solely for pragmatic reasons, it is
   not computationally feasible to run thousands of permutations for
   each one.)
-  **association test map**: z-scores from a two-way ANOVA testing for
   the presence of a non-zero association between term use and voxel
   activation. The association test maps provides somewhat different
   (and, in our view, typically more useful) information. Whereas the
   uniformity test maps tell you about the consistency of activation for
   a given term, the association test maps tell you whether activation
   in a region occurs more consistently for studies that mention the
   current term than for studies that don't. So for instance, the fact
   that the amygdala shows a large positive z-score in the association
   test map for emotion implies that studies whose abstracts include the
   word 'emotion' are more likely to report amygdala activation than
   studies whose abstracts don't include the word 'emotion'. That's
   important, because it controls for base rate differences between
   regions. Meaning, some regions (e.g., dorsal medial frontal cortex
   and lateral PFC) play a very broad role in cognition, and hence tend
   to be consistently activated for many different terms, despite
   lacking selectivity. The association test maps let you make slightly
   more confident claims that a given region is involved in a particular
   process, and isn't involved in just about every task.

Nimare
----------------------------------------------------------------

Coordinate based meta-analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
