"""
Copyright (C) 2019, Christian Donner

This file is part of SGCP_Inference.

SGCP_Inference is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SGCP_Inference is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SGCP_Inference.  If not, see <http://www.gnu.org/licenses/>.
"""
__author__ = 'Christian Donner'
__email__ = 'christian.donner(at)bccn-berlin.de'
__license__ = 'gpl-3.0'

import numpy

def sample_gp(x, cov_params):
    """ Samples a GP.

    :param x: numpy.ndarray [num_points x dims]
        Points, where the GP is sampled at.
    :param cov_params: list
        Parameters of covariance function.

    :return: numpy.ndarray [num_points]
        Function values of GP a the points.
    """
    num_points, D = x.shape
    K = cov_func(x, x, cov_params)
    L = numpy.linalg.cholesky(K + 1e-8 * numpy.eye(num_points))
    rand_nums = numpy.random.randn(num_points)
    g = numpy.dot(L, rand_nums)

    return g

def cov_func(x, x_prime, cov_params):
    """ Computes the covariance functions between x and x_prime.

    :param x: numpy.ndarray [num_points x D]
        Contains coordinates for points of x
    :param x_prime: numpy.ndarray [num_points_prime x D]
        Contains coordinates for points of x_prime
    :param cov_params: list
        First entry is the amplitude, second an D-dimensional array with
        length scales.

    :return: numpy.ndarray [num_points x num_points_prime]
        Kernel matrix.
    """

    theta_1, theta_2 = cov_params[0], cov_params[1]

    x_theta2 = x / theta_2
    xprime_theta2 = x_prime / theta_2
    h = numpy.sum(x_theta2 ** 2, axis=1)[:, None] - 2. * numpy.dot(
        x_theta2, xprime_theta2.T) + \
        numpy.sum(xprime_theta2 ** 2, axis=1)[None]
    return theta_1 * numpy.exp(-.5*h)

def inhomogeneous_poisson_process(S_borders, lmbda_star, cov_params,
                                  grid_points=50, seed=None, num_sets=1):
    """ Samples an inhomogeneous Poisson process via thinning.

    :param S_borders: numpy.ndarray [D x 2]
        Limits of Region of Poisson process.
    :param lmbda_star: float
        Maximal rate/intensity of Poisson process
    :param cov_params: list
        Covariance parameters. First entry is the amplitude, second an
        D-dimensional array with length scales.
    :param grid_points:
        How many grid points per dimension should be used. (Careful for higher
        dimensions)

    :return:
        numpy.ndarray [num_pp_events x D]: Location of PP events
        numpy.ndarray [num_pp_events]: Intensity at PP events
        numpy.ndarray [num_grid_points x D]: Grid locations in each dimension
        numpy.ndarray: Intensity values at each grid point.
    """

    if seed is not None:
        numpy.random.seed(seed)
    D = S_borders.shape[0]
    S = S_borders[:, 1] - S_borders[:, 0]
    R = numpy.prod(S)
    expected_num_events = R * lmbda_star * num_sets
    num_events = numpy.random.poisson(expected_num_events, 1)[0]
    print(num_events)
    X_unthinned = numpy.random.rand(num_events, D)*S[None]
    event_set_id = numpy.random.randint(0, num_sets, num_events)

    X_grid = numpy.empty([grid_points, D])
    for di in range(D):
        X_grid[:, di] = numpy.linspace(0, S[di], grid_points)

    X_mesh = numpy.meshgrid(*X_grid.T.tolist())
    X_mesh = numpy.array(X_mesh).reshape([D, -1]).T
    X_points = numpy.concatenate([X_mesh, X_unthinned])
    g = sample_gp(X_points, cov_params)
    intensity_function = lmbda_star / (1. + numpy.exp(-g))
    thinned_idx = numpy.where(intensity_function[grid_points ** D:] >=
                              lmbda_star *
                              numpy.random.rand(len(X_unthinned)))[0]

    X_thinned = X_points[grid_points ** D:][thinned_idx]
    X_thinned += S_borders[:, 0][numpy.newaxis]
    intensity_thinned = intensity_function[grid_points ** D:][thinned_idx]

    X_sets = []
    intensity_sets = []
    event_set_id_thinned = event_set_id[thinned_idx]

    for iset in range(num_sets):
        X_set_idx = numpy.where(numpy.equal(event_set_id_thinned, iset))[0]
        X_sets.append(X_thinned[X_set_idx])
        intensity_sets.append(intensity_thinned[X_set_idx])

    intensity_array = \
        intensity_function[:grid_points ** D].reshape([grid_points] * D)
    X_grid += S_borders[:, 0][numpy.newaxis]

    return X_sets, intensity_sets, X_grid, intensity_array


def inhomogeneous_poisson_process_approx(S_borders, lmbda_star, cov_params,
                                         grid_points=50, seed=None, num_sets=1):
    """ Samples an inhomogeneous Poisson process via thinning. This is an
    approximate method for sampling realisations with many observations.
    First evaluates the intensity function on a dense grid, and then uses the
    values on the grid for thinning.

    :param S_borders: numpy.ndarray [D x 2]
        Limits of Region of Poisson process.
    :param lmbda_star: float
        Maximal rate/intensity of Poisson process
    :param kind_cov: str
        Indicates covariance function of GP.
    :param cov_params: list
        Covariance parameters
    :param grid_points:
        How many grid points per dimension should be used. (Careful for high
        dimensions)

    :return:
        numpy.ndarray [num_pp_events x D]: Location of PP events
        numpy.ndarray [num_grid_points x D]: Grid locations in each dimension
        numpy.ndarray: Intensity values at each grid point.
    """

    if seed is not None:
        numpy.random.seed(seed)
    D = S_borders.shape[0]
    S = S_borders[:, 1] - S_borders[:, 0]
    R = numpy.prod(S)
    expected_num_events = R * lmbda_star

    X_grid = numpy.empty([grid_points, D])
    for di in range(D):
        X_grid[:, di] = numpy.linspace(0, S[di], grid_points)

    X_mesh = numpy.meshgrid(*X_grid.T.tolist())
    X_mesh = numpy.array(X_mesh).reshape([D, -1]).T
    g = sample_gp(X_mesh, cov_params, kind_cov)
    intensity_function = lmbda_star / (1. + numpy.exp(-g))

    X_sets = []

    for iset in range(num_sets):
        num_events = numpy.random.poisson(expected_num_events, 1)[0]
        print(num_events)
        X_unthinned = numpy.random.rand(num_events, D) * S[None]
        rand_nums = lmbda_star * numpy.random.rand(len(X_unthinned))
        thinned_idx = []
        for ipoint in range(X_unthinned.shape[0]):
            X = X_unthinned[ipoint]
            distance = numpy.sqrt(numpy.sum((X_mesh - X[numpy.newaxis])**2, axis=1))
            g_point_idx = numpy.argmin(distance)
            if intensity_function[g_point_idx] >= rand_nums[ipoint]:
                thinned_idx.append(ipoint)

        X_sets.append(X_unthinned[numpy.array(thinned_idx),:])


    intensity_array = intensity_function.reshape([grid_points] * D)
    X_grid += S_borders[:, 0][numpy.newaxis]

    return X_sets, X_grid, intensity_array