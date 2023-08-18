The brain tractograms were computed for every subject based on fODFs maps and a probabilistic tractography method. However, 2 different algorithms with different parameters were used:

# Determinstic
# Probabilistic
## Particle filter tracking (PFT)
PFT is implemented directly in _TractoFlow_ using `dipy.tracking.local_tracking.ParticleFilteringTracking` ([tutorial](https://dipy.org/documentation/1.5.0/examples_built/tracking_pft/#example-tracking-pft)) and is based on partial volume estimation (PVE) maps to only reconstruct streamlines that connect two GM areas. In our case, the tractogram is computed using interface seeding and 30 seeds/voxel for connectivity and streamline density analysis.

## Local tracking
Local tracking is implemented directly in _TractoFlow_ using `dipy.tracking.local_tracking.LocalTracking` ([tutorial](https://dipy.org/documentation/1.5.0/examples_built/tracking_probabilistic/#example-tracking-probabilistic)) and is based on a FA mask (FA threshold = 0.05) to reconstruct most possible streamlines. In our case, the tractogram is computed using an FA mask with a threshold of 0.05 and 5 seeds/voxel for tractometry and whole bundle analysis.