# vsr-behavior-characterization
In this small project we apply Unsupervised Learning techniques to find
recurrent behaviors in a dataset of Voxel-Based Soft Robots (VSR).

## Structure of the project
The folder `src` contains the core of the project, in particular the file
`behavior_classification.py` is responsible for the execution of the
experiments. The other files are auxiliary functions/classes used to clean
the code.

The folder `dataset` contains the data which we arw willing to clusterize.
`best.[0-9].txt` contain the predictors on which we apply KMeans, please refer to
[this](https://medvet.inginf.units.it/teaching/2122-intro-ml-er/project/#vsrs-behavior-characterization)
webpage for an overview of their meaning. The files `supervised_clusters*.py`
contain two different versions of verification datasets that we prepared by
hand via the manual inspection of videos available alongside the datasets.

## Example run and output
TBD

## Authors
+ Francesco Andreuzzi
+ Luca Filippi

## References
1. Ferigo et al. 2021, *Beyond body shape and brain: evolving the sensory apparatus of voxel-based soft robots.*
2. Hastie, Tibshirani, Friedman, 2009, *An introduction to statistical learning.*
3. Medvet et al. 2020, *Design, validation, and case studies of 2d-vsr-sim, an optimization-friendly simulator of 2-d voxel-based soft robots.*
4. Medvet et al. 2021, *Biodiversity in evolved voxel-based soft robots.*
5. Panday et al. 2018, *Feature weighting as a tool for unsupervised feature selection.*
