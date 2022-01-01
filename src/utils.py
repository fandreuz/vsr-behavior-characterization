import numpy as np
from itertools import permutations


def beautiful_padded(key, value, n_pad=40):
    return "{}{}{}".format(
        key,
        "".join([" " for _ in range(n_pad - len(key))]),
        value,
    )


def report_results(
    clusters, clusters_label, error, error_details, mapping, truth_clusters
):
    print(
        "The best combination ({}) leads to {}/{} errors".format(
            clusters_label,
            error,
            sum(map(lambda s: len(s._items), truth_clusters)),
        )
    )
    print("\n------- Number of misclassified robots")
    for i in range(len(truth_clusters)):
        print(
            "- {} robots wrongly put into the (supervised) cluster {};".format(
                error_details[i], truth_clusters[i].name
            )
        )
    print("\n------- Size of unsupervised clusters")
    print("Mapping: " + str(mapping))
    for i in range(len(truth_clusters)):
        print(
            "- The supervised cluster {} got {} out of {} expected robots;".format(
                truth_clusters[i].name,
                clusters[mapping[i]].size(),
                truth_clusters[i].size(),
            )
        )
    print("\n------- Clusters intersection table")
    clusters_intersection_table(
        clusters,
        "Unsuper",
        truth_clusters,
        "Super",
        len(clusters),
    )


def clusters_difference_1vsmany(one, many):
    return " ".join(
        map(
            lambda s: s.ljust(3),
            map(
                str,
                [len(one.difference(many[j])) for j in range(len(many))],
            ),
        )
    )


def clusters_intersection_1vsmany(one, many):
    return " ".join(
        map(
            lambda s: s.ljust(3),
            map(
                str,
                [len(one.intersect(many[j])) for j in range(len(many))],
            ),
        )
    )


def clusters_intersection_table(
    cls1, cls1_label, cls2, cls2_label, n_clusters, n_pad=10
):
    print(" " * (n_pad + 3) + cls2_label.center(n_clusters * 2))
    print(
        " " * (n_pad + 3)
        + " ".join(map(lambda s: s.ljust(3), map(str, range(n_clusters))))
    )
    print(" " * (n_pad + 3) + "-" * (n_clusters * 3 + n_clusters - 1))
    for i in range(n_clusters):
        if i == n_clusters // 2:
            print(
                cls1_label.center(n_pad)
                + str(i)
                + "| "
                + clusters_intersection_1vsmany(cls1[i], cls2)
            )
        else:
            print(
                " " * n_pad
                + str(i)
                + "| "
                + clusters_intersection_1vsmany(cls1[i], cls2)
            )


# return the error which results from the j-th supervised cluster (columns of
# intersections_matrix) to the mapping[j]-th unsupervised cluster (rows of
# intersections_matrix)
def compute_error(mapping, intersections_matrix):
    for i in range(len(mapping)):
        intersections_matrix[mapping[i], i] = 0
    return np.sum(intersections_matrix), np.sum(intersections_matrix, axis=0)


# compute the error committed in clusters, which is defined as the sum of the
# elements on each row that are not the row-wise maximum
# Example:
#
# 0  1  5
# 1  1  7
# 10 0  0
#
# the error in this case is 3
# returns a tuple containing:
# 1. total count of errors
# 2. count of misclassified robots for each supervised cluster
# 3. mapping from supervised cluster to the corresponding unsupervised (via
#       indexes)
def clusters_comparison(clusters, supervised_cls):
    intersections = np.zeros((len(clusters), len(supervised_cls)), dtype=int)
    for i in range(len(clusters)):
        for j in range(len(supervised_cls)):
            intersections[i, j] = len(clusters[i].intersect(supervised_cls[j]))

    best_total_errors = 100000
    best_super_errors = None
    best_mapping = None

    for mapping in permutations(range(len(clusters))):
        ttl_err, spr_err = compute_error(mapping, np.array(intersections))
        if ttl_err < best_total_errors:
            best_total_errors = ttl_err
            best_super_errors = spr_err
            best_mapping = mapping

    return best_total_errors, best_super_errors, best_mapping
