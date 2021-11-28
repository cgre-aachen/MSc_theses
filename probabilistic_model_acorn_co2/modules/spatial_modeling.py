"""
    This file is part of AMTK - the Acorn mesh toolkit, the companion
    software package to the Master's Thesis "The value of probabilistic
    geological modeling: Application to the Acorn CO2 storage site"
    of Marco van Veen

    AMTK is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AMTK is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with AMTK.  If not, see <http://www.gnu.org/licenses/>.

@author: Marco van Veen
"""

import numpy as np
from scipy.spatial.distance import cdist

### semivariogram model functions
### First three from: Oliver, M. A., & Webster, R. (2014). A tutorial guide to geostatistics: Computing and modelling variograms and kriging. CATENA, 113, 56–69. https://doi.org/10.1016/j.catena.2013.09.006


def vario_spher(h, a, c, c0):
    '''  Isotropic spherical-plus-nugget model function
    a: range
    h: lag
    c: spatially correlated variance
    c0: nugget variance '''

    h = np.abs(np.array(h))  # np.piecewise is a bit picky with lists
    gamma = np.piecewise(h, [(0 < h) & (h <= a),
                             h > a,
                             h == 0],
                         [lambda h: c0 + c * ((3 * h) / (2 * a) - 0.5 * (h / a) ** 3),
                          lambda h: c0 + c,
                          lambda h: 0]
                         )
    return gamma


def vario_exp(h, a, c, c0):
    ''' Exponential variogram model function
    a: range (effective range is approx. 3*a)
    h: lag
    c: spatially correlated variance
    c0: nugget variance '''

    h = np.abs(h)
    gamma = np.where(h > 0,
                     c0 + c * (1 - np.exp(-h / a)),
                     0)
    return gamma


def vario_gau(h, a, c, c0):
    ''' Gaussian variogram model function
    a: range (effective range is approx. sqrt(3*a))
    h: lag
    c: spatially correlated variance
    c0: nugget variance '''

    h = np.abs(h)
    gamma = np.where(h > 0,
                     c0 + c * (1 - np.exp(-h ** 2 / a ** 2)),
                     0)
    return gamma


def vario_power(h, b, m, c0):
    ''' Power variogram model function
    h: lag
    b: multiplier
    m: exponent (0 < m < 2)
    c0: nugget variance '''

    h = np.abs(h)
    gamma = np.where(h > 0,
                     c0 + b * h ** m,
                     0)
    return gamma


def vario_lin(h, a, c, c0):
    ''' Linear variogram model function
    h: lag
    c: spatially correlated variance
    c0: nugget variance '''

    h = np.abs(h)
    gamma = np.where(h < a,
                     c0 + h * ((c - c0) / a),
                     c)
    return gamma

### Covariance model functions
## covariance functions derived from modelled variograms
#
# Arguments:
#     h: lag
#     a: range
#     c: spatially correlated variance
#     c0: nugget variance
# Return:
#     gamma: covariance value
#
#
# def cov_spher(h, a, c, c0):
#     gamma = np.where(abs(h) <= a,
#                      c * (1 - 3 / 2 * abs(h) / a + 1 / 2 * abs(h) ** 3 / a ** 3),
#                      0.)
#     return gamma
#
# def cov_exp(h, a, c, c0):
#     gamma = c * (np.exp(- abs(h) / a))
#     return gamma
#
# def cov_gauss(h, a, c, c0):
#     gamma = c * (np.exp(- abs(h) / a) ** 2)
#     return gamma
#
# def cov_lin(h, a, c, c0):
#     gamma = - h * (c / a) + c
#     return gamma


def ordinary_kriging_var(h, H, prop, var_function, sgs=False, **kwargs):
    # ordinary kriging, using semivariance function, single estimation point / sgs
    ### modified from Jan von Harten ###
    '''
    Method for simple kriging calcualtion.
    Args:
        h: distance matrix containing all distances between target point and moving neighbourhood
        H: distance matrix containing all inter-point distances between locations in moving neighbourhood
        prop: array containing scalar property values of locations in moving neighbourhood

        var_function: semivariance function
        **kwargs: values to be passed to semivariance function (e.g. range, sill, nugget)
        sgs: If True, estimate value will be randomly inside the kriging standard deviation
    Returns:
        result: single scalar property value estimated for target location
    '''

    # empty matrix building
    l = h.shape[0]
    C = np.ones((l + 1, l + 1))
    c = np.ones((l + 1))

    # Filling matrices with covariances based on calculated distances and exponential variogram model
    C[:l, :l] = var_function(H, **kwargs)
    c[:l] = var_function(h, **kwargs)

    # Adding ordinary constraints on weights
    np.fill_diagonal(C, 0)

    # calculate weights by solving system
    w = np.inner(np.linalg.inv(C), c)

    # calculate estimate
    estimate = np.inner(prop, w[:-1])

    # calculate ordinary kriging variance
    variance = w[-1] + np.inner(c[:-1], w[:-1])  # CHECKED

    if sgs:
        # SGS version - taking result from normal distribution defined by kriging estimate an standard deviation
        # np.random.seed(random_seed)
        estimate = np.random.normal(estimate, np.sqrt(np.abs(variance)))

    return estimate, variance


def ordinary_kriging_var_batch(est, dat, var_function, **kwargs):
    # ordinary kriging, using semivariance function, batch estimation
    """
    Using array combinations instead of loops to calculate several estimation points ***

    Arguments:
        est: size of the estimation grid
        data_values: arrays of x and y positions of known data values + array of measurements z (according to positions)

        var_function: semivariance function
        **kwargs: values to be passed to semivariance function (e.g. range, sill, nugget)
    """

    el = est.shape[0]
    dl = dat.shape[0]

    # calculate distances/lag between positions of data values and the estimation positions
    h = cdist(est[:, :2], dat[:, :2])

    # calculate distances/lag between possible data value combinations
    H = cdist(dat[:, :2], dat[:, :2])

    # create matrices
    C = np.ones((dl + 1, dl + 1))
    c = np.ones((el, dl + 1))

    # calculate covariances
    C[:-1, :-1] = var_function(H, **kwargs)
    c[:, :-1] = var_function(h, **kwargs)

    # important for semivariance
    np.fill_diagonal(C, 0)

    # solve the ordinary kriging system C * w = c for weigths w
    w = np.inner(np.linalg.inv(C), c).T

    # calculate estimate
    estimates = np.inner(dat[:, 2], w[:, :-1])

    # calculate ordinary kriging variance
    variances = w[:, -1] + np.sum((c[:, :-1] * w[:, :-1]), axis=1)  # CHECKED

    return estimates, variances


def ordinary_kriging_cov(h, H, prop, cov_function, sgs=False, **kwargs):
    # ordinary kriging, using covariance function, single estimation point / sgs
    '''
    Method for simple kriging calcualtion.
    Args:
        h: distance matrix containing all distances between target point and moving neighbourhood
        H: distance matrix containing all inter-point distances between locations in moving neighbourhood
        prop: array containing scalar property values of locations in moving neighbourhood

        cov_function: covariance function
        **kwargs: values to be passed to covariance function (e.g. range, sill, nugget)
        sgs: If True, estimate value will be randomly inside the kriging standard deviation
    Returns:
        result: single scalar property value estimated for target location
    '''

    # empty matrix building / adding constraints on weights
    l = h.shape[0]
    c = np.ones((l + 1))
    C = np.ones((l + 1, l + 1))
    C[l, l] = 0

    # calculate covariances
    C[:l, :l] = cov_function(H, **kwargs)
    c[:l] = cov_function(h, **kwargs)

    # solve the ordinary kriging system C * w = c for weigths w
    w = np.inner(np.linalg.inv(C), c)

    # calculate estimate
    estimate = np.inner(prop, w[:-1])

    # calculate ordinary kriging variance
    variance = C[0, 0] - np.inner(c, w)  # CHECKED

    if sgs:
        # simulate result from normal distribution defined by kriging estimate an standard deviation
        # np.random.seed(random_seed)
        estimate = np.random.normal(estimate, np.sqrt(np.abs(variance)))

    return estimate, variance


def ordinary_kriging_cov_batch(est, dat, cov_function, **kwargs):
    # ordinary kriging, using covariance function, batch estimation
    """
    Using array combinations instead of loops to calculate several estimation points ***

    Arguments:
        est: size of the estimation grid
        data_values: arrays of x and y positions of known data values + array of measurements z (according to positions)

        cov_function: covariance function
        **kwargs: values to be passed to covariance function (e.g. range, sill, nugget)
        sgs: If True, estimate value will be randomly inside the kriging standard deviation
    """

    el = est.shape[0]
    dl = dat.shape[0]

    # calculate distances/lag between positions of data values and the estimation positions
    h = cdist(est[:, :2], dat[:, :2])

    # calculate distances/lag between possible data value combinations
    H = cdist(dat[:, :2], dat[:, :2])

    # extending matrices to perform ordinary kriging
    C = np.ones((dl + 1, dl + 1))
    C[-1, -1] = 0

    c = np.ones((el, dl + 1))

    # calculate covariances
    C[:-1, :-1] = cov_function(H, **kwargs)
    c[:, :-1] = cov_function(h, **kwargs)

    # solve the ordinary kriging system C * w = c for weigths w
    w = np.inner(np.linalg.inv(C), c).T

    # calculate estimate
    estimates = np.inner(dat[:, 2], w[:, :-1])

    # calculate ordinary kriging variance
    variances = C[0, 0] - np.sum((c * w), axis=1)  # CHECKED

    return estimates, variances


def ordinary_sgs_var(est, dat, var_function, distance_points=20, seed=None, **kwargs):
    # get sizes
    dl = dat.shape[0]
    el = est.shape[0]

    # initiate results
    res = np.empty(el)

    # initiate simulation data
    sim = np.zeros([dl + el, 3])
    sim[:dl] = dat

    pos = np.arange(el)
    np.random.seed(seed=seed)
    np.random.shuffle(pos)

    for i, idx in enumerate(pos):
        # select current simulation points by i
        current_sim = sim[:dl + i]

        h = cdist([est[idx, :2]], current_sim[:, :2])[0]

        # creating distance masks to slice arrays regarding a certain number of nearest points
        dm = np.argsort(h)[:distance_points]
        H = cdist(current_sim[dm, :2], current_sim[dm, :2])

        e = ordinary_kriging_var(h[dm], H, current_sim[dm, 2],
                                    var_function=var_function,
                                    sgs=True,
                                    **kwargs)[0]

        res[idx] = e
        sim[dl + i, :2] = est[idx, :2]
        sim[dl + i, 2] = e

    return res


def ordinary_sgs_pilot_var(est, dat, var_function, pilot_points, distance_points=20, seed=None, **kwargs):
    el = est.shape[0]

    np.random.seed(seed=seed)
    pos = np.random.randint(0, el, size=pilot_points)

    pilots = np.zeros((pilot_points, 3))
    pilots[:, :2] = est[pos]

    pilots[:, 2] = ordinary_sgs_var(pilots[:, :2], dat,
                                    var_function,
                                    distance_points=distance_points,
                                    seed=seed,
                                    **kwargs)

    dp = np.vstack([dat, pilots])

    val = ordinary_kriging_var_batch(est, dp, var_function, **kwargs)[0]

    return val, pilots


def get_pilot_points(est, dat, n, distance, seed=None):
    '''
    est: Set of possible pilot points, can be the estimation grid
    dat: points of the same kin, which will be used to define a distance filter
    n: desired number of pilot points, not guaranteed to be reached
    distance: distance or range, pilot points should be away from dat points'''

    np.random.seed(seed=seed)
    pilots = np.zeros((n, 3))

    # create empty array for dat points and pilot points
    dl = dat.shape[0]
    sim = np.zeros([dl + n, 3])
    sim[:dl] = dat

    i = 0  # counter
    while i < n:

        # calculate all distances between remaining candidates and sim points
        dist = cdist(sim[:dl + i, :2], est[:, :2])
        # choose candidates which are out of range
        mm = np.min(dist, axis=0)
        candidates = est[mm > distance]
        # count candidates
        cl = candidates.shape[0]
        if cl < 1: break
        # randomly pick candidate and set next pilot point
        pos = np.random.randint(0, cl)
        pilots[i, :2] = candidates[pos]
        sim[dl + i, :2] = candidates[pos]

        i += 1

    # just return valid points if early break occured
    pilots = pilots[:i]

    return pilots


def ordinary_pilot_var(est, dat, var_function, pilots_n, pilots_distance, value_generator, seed=None, **kwargs):
    pilots = get_pilot_points(est, dat, pilots_n, pilots_distance)

    # random value sampling, only for returned points (can be smaller than pilots_n)!
    pilots[:, 2] = value_generator(pilots.shape[0])

    dp = np.vstack([dat, pilots])

    val = ordinary_kriging_var_batch(est, dp, var_function, **kwargs)[0]

    return val, pilots