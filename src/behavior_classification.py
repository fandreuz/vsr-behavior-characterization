from utils import beautiful_padded

import pandas as pd
import numpy as np
import sys
import csv

from sklearn.cluster import KMeans

# ----- Interpreting Gait -----
# gait→avg.touch.area : M* is an n-gram of footprints m_i. touch_area is the
#                       number of True elements in m_i over all the footprints
#                       in the n-gram, divided by 8*n (8 is the number of
#                       components of each footprint).
# gait→num.footprints': n, i.e. the number of footprints in the n-gram.
# gait→mode.interval' : Since each footprint is collected at a fixed time
#                       interval DeltaT = 0.5l, the distance in time between
#                       two occurrences of the main n-gram is a discrete
#                       variable. Therefore we may compute its mode.
# gait→coverage'      : ???
# gait→purity'        : Rate of time intervals between two consecutive
#                       occurrences of the main n-gram which are equal to the
#                       mode gait→mode.interval.
# gait→num.unique.footprints' : Number of unique footprints in the n-gram.
# gait→footprints'    : Comma-separated list of footprints in the main n-gram.

if len(sys.argv) > 1:
    clusters = int(sys.argv[1])
else:
    clusters = 3

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

X = np.concatenate(
    [
        training_data[i][training_data[i]["iterations"] == 204][columns]
        for i in range(len(training))
    ]
)

kmeans = KMeans(n_clusters=clusters, algorithm="full").fit(X)

# these are our clusters
labels = kmeans.labels_

dc = {}

l = X.shape[0] // len(training)
for i in range(len(training)):
    for row, label in zip(
        training_data[i][training_data[i]["iterations"] == 204].itertuples(),
        labels[l * i : l * (i + 1)],
    ):
        key = (row.terrain, row.shape)
        if key not in dc:
            dc[key] = []
        dc[key].append(str(label))

print(beautiful_padded("SEED", " ".join(map(str, range(10)))))
print("-" * 40 + "-" * 2 * len(training_data))
for k, l in dc.items():
    print(beautiful_padded(str(k) + " ->    ", " ".join(l)))

with open("dataset/clusters.csv", "w") as f:
    writer = csv.writer(f)
    for t, s in dc.keys():
        lb = dc[(t, s)]
        row = [t, s, *lb]
        writer.writerow(row)
