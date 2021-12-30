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
