import numpy as np
from cluster import Cluster, VSR
from sklearn.cluster import KMeans

random_samples = False

# run an unsupervised learning classification using the given X. return the
# clusters found in the experiment
def run_experiment(X, training_data, weights, n_clusters):
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
