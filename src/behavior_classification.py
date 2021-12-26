import pandas as pd
import numpy as np
import sys
import csv

from sklearn.cluster import KMeans

n_pad = 40

def beautiful_padded(key, value):
    return "{}{}{}".format(
                key,
                "".join([" " for _ in range(n_pad - len(key))]),
                value,
            )

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

if len(sys.argv) > 1:
    clusters = int(sys.argv[1])
else:
    clusters = 3

kmeans = KMeans(n_clusters=clusters).fit(X)
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


print(beautiful_padded('SEED', ' '.join(map(str, range(10)))))
for k, l in dc.items():
    print(beautiful_padded(str(k) + ' ->    ', ' '.join(l)))

with open('dataset/clusters.csv','w') as f:
    writer = csv.writer(f)
    for t, s in dc.keys():
        lb = dc[(t,s)]
        row = [t,s,*lb]
        writer.writerow(row)
