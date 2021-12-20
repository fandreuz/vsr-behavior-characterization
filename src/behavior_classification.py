import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

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

training = [7, 8, 9]
training_data = [
    pd.read_csv("dataset/best.{}.txt".format(i), sep=";") for i in training
]

X = np.concatenate(
    [
        training_data[i][training_data[i]["iterations"] == 204][columns]
        for i in range(3)
    ]
)

# we expect three clusters (walking, jumping, crawling)
kmeans = KMeans(n_clusters=3).fit(X)
labels = kmeans.labels_

l = X.shape[0] // 3
for i in range(3):
    print('------------------ best.{}.txt'.format(training[i]))
    for row, label in zip(
        training_data[i][training_data[i]["iterations"] == 204].itertuples(),
        labels[l * i : l * (i + 1)],
    ):
        descriptive_label = '[{},{}]'.format(row.terrain, row.shape)
        print('{} -> {} {}'.format(descriptive_label, ''.join([' ' for _ in range(30 - len(descriptive_label))]), label))
