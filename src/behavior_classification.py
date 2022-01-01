from utils import (
    clusters_comparison,
    clusters_intersection_table,
    report_results,
)
from indexes import *
from cluster import Cluster, VSR

import pandas as pd
import numpy as np
from itertools import product

import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir + "/dataset")

from supervised_clusters import *

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----- Interpreting Gait -----
# gait->avg.touch.area : M* is an n-gram of footprints m_i. touch_area is the
#                       number of True elements in m_i over all the footprints
#                       in the n-gram, divided by 8*n (8 is the number of
#                       components of each footprint).
# gait->num.footprints': n, i.e. the number of footprints in the n-gram.
# gait->mode.interval' : Since each footprint is collected at a fixed time
#                       interval DeltaT = 0.5l, the distance in time between
#                       two occurrences of the main n-gram is a discrete
#                       variable. Therefore we may compute its mode.
# gait->coverage'      : ???
# gait->purity'        : Rate of time intervals between two consecutive
#                       occurrences of the main n-gram which are equal to the
#                       mode gait->mode.interval.
# gait->num.unique.footprints' : Number of unique footprints in the n-gram.
# gait->footprints'    : Comma-separated list of footprints in the main n-gram.


# run an unsupervised learning classification using the given X. return the
# clusters found in the experiment
def run_experiment(X, training_data):
    # unsupervised learning engine
    kmeans = KMeans(n_clusters=n_clusters, algorithm="full").fit(
        (X * weights)[:, random_columns]
    )

    # these are our clusters
    labels = kmeans.labels_

    clusters = tuple(Cluster(name=str(i)) for i in range(n_clusters))

    l = X.shape[0] // len(training_data)
    # we iterate over the files (i.e. the seed)
    for i in range(len(training_data)):
        # for each row we want to extract terrain/shape, and the cluster that
        # robot was assigned to
        for row, label in zip(
            training_data[i][
                training_data[i]["iterations"] == 204
            ].itertuples(),
            labels[l * i : l * (i + 1)],
        ):
            # we add the VSR to the cluster which it was assigned to
            clusters[label].add(
                VSR(shape=row.shape, training_terrain=row.terrain, seed=i)
            )

    return clusters


if len(sys.argv) > 1:
    n_clusters = int(sys.argv[1])
else:
    n_clusters = 3

if len(sys.argv) > 2:
    random_samples = bool(int(sys.argv[2]))
else:
    random_samples = False

normalize_spectra = True

# ------------------------------------------------------
# allocate and fill supervised clusters
supervised_clusters = []

for label, bp, wm in zip(
    ["Crawling", "Walking", "Jumping", "Rolling"],
    [biped_crawling, biped_walking, biped_jumping, biped_rolling],
    [worm_crawling, worm_walking, worm_jumping, worm_rolling],
):
    if n_clusters > 3 or label != "Rolling":
        cl = Cluster(name=label)
        supervised_clusters.append(cl)

        for item in bp:
            cl.add(
                VSR(shape="biped-4x3", training_terrain=item[0], seed=item[1])
            )
        for item in wm:
            cl.add(
                VSR(shape="worm-5x2", training_terrain=item[0], seed=item[1])
            )
# ------------------------------------------------------

# we define some mappings (i.e. weights) for the predictor 'avg.touch.area'
avg_touch_area_mappings = []
avg_touch_are_labels = []

coeffs_grid = [0, 0.1, 0.2, 0.5, 0.75, 1, 1.5, 5, 10]
for mi, nfoot, nuniquefoot, purity_degree in product(
    coeffs_grid, coeffs_grid, coeffs_grid, [0, 1, 2, 3, 4, 5]
):

    def mppng(
        X,
        mi=mi,
        nfoot=nfoot,
        nuniquefoot=nuniquefoot,
        pd=purity_degree,
    ):
        return (
            mi * k_mode_interval(X) * k_purity(X, degree=pd)
            + nfoot * k_nfootprints(X)
            + nuniquefoot * k_unique_foorprints(X)
        )

    avg_touch_area_mappings.append(mppng)
    avg_touch_are_labels.append(
        "mode_interval_cff={}, n_footprints_cff={}, n_unique_footprints_cff={}, purity_degree={}".format(
            mi, nfoot, nuniquefoot, purity_degree
        )
    )
# ------------------------------------------------------


# we consider these two set of columns the key predictors
columns = [
    "best→fitness→as[Outcome]→center.spectrum.x→[{}]".format(i)
    for i in range(10)
]
columns.extend(
    [
        "best→fitness→as[Outcome]→center.spectrum.y→[{}]".format(i)
        for i in range(10)
    ]
)

# load the dataset
training = list(range(10))
training_data = [
    pd.read_csv("dataset/best.{}.txt".format(i), sep=";") for i in training
]
concatenated_training_data = pd.concat(
    data[data["iterations"] == 204] for data in training_data
)

# predictors used for unsupervised learning
X = concatenated_training_data[columns].values

# normalize spectrum
if normalize_spectra:
    X -= np.mean(X)

    vr = np.std(X)
    if vr == 0:
        vr = 1
    X /= vr

# add space for average touch area
X = np.hstack([np.zeros((X.shape[0], 1)), X])

# ------------------------------------------------------

avg_touch_area_weights = [1, 2, 5, 10, 100]

# ------------------------------------------------------

ata_key = "best→fitness→as[Outcome]→gait→avg.touch.area"

# an array of arrays of clusters
experiment_clusters = []
experiment_errs = []
experiment_err_details = []
experiment_labels = []
# map supervised cluster to the corresponding unsupervised cluster
experiment_super_to_unsuper = []

for avg_touch_mapping, avg_touch_label in zip(
    avg_touch_area_mappings, avg_touch_are_labels
):
    for avg_touch_area_w in avg_touch_area_weights:
        experiment_label = (
            avg_touch_label
            + ", avg_touch_area weight={}".format(avg_touch_area_w)
        )

        print("-" * len(avg_touch_label))
        print(experiment_label)

        weighted_avg_touch = concatenated_training_data[
            ata_key
        ] * avg_touch_mapping(concatenated_training_data)
        X[:, 0] = weighted_avg_touch

        # setup KMeans weights
        weights = np.ones(X.shape[1])
        weights[0] = avg_touch_area_w

        if random_samples:
            random_columns = np.concatenate(
                [
                    [0, 1],
                    np.random.randint(2, X.shape[1], (X.shape[1] - 2) // 2),
                ]
            )
        else:
            random_columns = np.arange(0, X.shape[1])
        print("Using the following columns: " + str(random_columns))

        clusters = run_experiment(X, training_data)
        experiment_clusters.append(clusters)

        err, err_details, mapping = clusters_comparison(
            clusters, supervised_clusters
        )
        print("Errors: " + str(err))

        experiment_errs.append(err)
        experiment_err_details.append(err_details)
        experiment_super_to_unsuper.append(mapping)
        experiment_labels.append(experiment_label)

experiment_errs = np.array(experiment_errs)
experiment_err_details = np.array(experiment_err_details)
experiment_super_to_unsuper = np.array(experiment_super_to_unsuper)

best = np.argmin(experiment_errs)

print("\n-------------------- RESULTS ---------------------")
report_results(
    experiment_clusters[best],
    experiment_labels[best],
    experiment_errs[best],
    experiment_err_details[best],
    experiment_super_to_unsuper[best],
    supervised_clusters,
)

print("\n------- Same error in: ")
for i in np.where(experiment_errs == experiment_errs[best])[0]:
    print(experiment_labels[i])

print("\n\n####### Error using unweighted average_touch_area #######")
X[:, 0] = concatenated_training_data[ata_key].values
clusters = run_experiment(X, training_data)
err, err_details, mapping = clusters_comparison(clusters, supervised_clusters)
report_results(
    clusters,
    "Unweighted avg_touch_area",
    err,
    err_details,
    mapping,
    supervised_clusters,
)
