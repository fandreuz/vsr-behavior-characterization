def beautiful_padded(key, value, n_pad=40):
    return "{}{}{}".format(
        key,
        "".join([" " for _ in range(n_pad - len(key))]),
        value,
    )
