# this module packs several indexes which we use as weights for other predictors

import numpy as np


# this is a measure of how much the most dominant footprints are important in
# the dynamics of the VSR. if the mode interval is big (with respect to the
# bigger mode interval of the dataset) we deduce that the main footprints are
# less important.
def k_mode_interval(X):
    longest_interval = np.max(X["best→fitness→as[Outcome]→gait→mode.interval"])
    return longest_interval / X["best→fitness→as[Outcome]→gait→mode.interval"]


# consider also the "purity" (how many intervals are equal to
# the mode interval?). obviously if purity=1, kmi is very significant. as
# purity->0 the significance decreases (the degree of decrease may vary, if
# the degree is high the decrease in significance is very fast).
def k_purity(X, degree=2):
    return np.power(X["best→fitness→as[Outcome]→gait→purity"], degree)


# if the number of footprints in the main gait is low, it means that the
# framework struggled to find recurrent patterns in the footprints. which means
# that the patterns it found (struggling) may be assumed to be less significant.
def k_nfootprints(X):
    longest_main_footprint = np.max(
        X["best→fitness→as[Outcome]→gait→num.footprints"]
    )
    return (
        X["best→fitness→as[Outcome]→gait→num.footprints"]
        / longest_main_footprint
    )


# if the number of unique footprints in a main footprint is high, this means
# that there's high variability in the main footprint, i.e. no clear info can
# be recovered. by contrast, if the number of unique footprints is low with
# respect to the number of footprints, this means that we captured important
# info regarding the behavior of the robot.
def k_unique_foorprints(X):
    return (
        X["best→fitness→as[Outcome]→gait→num.footprints"]
        / X["best→fitness→as[Outcome]→gait→num.unique.footprints"]
    )
