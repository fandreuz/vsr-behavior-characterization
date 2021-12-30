import numpy as np

def beautiful_padded(key, value, n_pad=40):
    return "{}{}{}".format(
        key,
        "".join([" " for _ in range(n_pad - len(key))]),
        value,
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


def clusters_comparison(
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

# compute the error committed in clusters, which is defined as the sum of the
# elements on each row that are not the row-wise maximum
# Example:
#
# 0  1  5
# 1  1  7
# 10 0  0
#
# the error in this case is 3
def clusters_error(clusters, supervised_cls):
    intersections = np.zeros((len(clusters), len(supervised_cls)), dtype=int)
    for i in range(len(clusters)):
        for j in range(len(supervised_cls)):
            intersections[i,j] = len(clusters[i].intersect(supervised_cls[j]))

    max_idxes = np.argmax(intersections, axis=1)
    for i in range(len(clusters)):
        intersections[i, max_idxes[i]] = 0
    return np.sum(intersections)
