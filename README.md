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
TBD
