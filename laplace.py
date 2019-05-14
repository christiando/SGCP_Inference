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
import time
from scipy.integrate import quadrature
from scipy.linalg import solve_triangular


class Laplace_SGCP():

    def __init__(self, S_borders, X, cov_params, num_inducing_points,
                 lmbda_star=None, conv_crit=1e-4,
                 num_integration_points=1000, output=False, noise=1e-4):
        """ Class initialisation for EM & Laplace inference for
        sigmoidal Gaussian Cox process. Note, that hyperparameter
        optimisation is not implemented.

        :param S_borders: numpy.ndarray [D x 2]
            Limits of the region of interest.
        :param X: numpy.ndarray [num_points x D]
            Positions of the observations.
        :param cov_params: numpy.ndarray [D + 1]
            Hyperparameters of the covariance functions. First is amplitude,
            and the others the length scale for each dimension.
        :param num_inducing_points: int
            Number of inducing points (Should be a power of dimensions)
        :param lmbda_star: float
            Maximal intensity. If None it is initialized as twice the mean
            observation rate for a homogeneous process. (Default=None)
        :param conv_crit:
            Convergence criterion, when algorithm should stop. (Default=1e-4)
        :param num_integration_points: int
            Number of points that should be used for Monte Carlo integration.
            (Default = 1000)
        :param output: bool
            Prints info after each optimisation step. (Default=False)
        :param noise: float
            Noise added to the diagonal of the covariance matrix (should be
            small). (Default=1e-4)
        """

        self.S_borders = S_borders
        self.S = S_borders[:,1] - S_borders[:,0]
        self.R = numpy.prod(self.S)
        self.D = S_borders.shape[0]
        self.cov_params = cov_params
        self.num_integration_points = num_integration_points
        self.num_inducing_points = num_inducing_points  # must be power of D
        self.noise = noise
        self.X = X

        self.place_inducing_points()
        self.g_s = numpy.zeros(self.induced_points.shape[0])
        self.Ks = self.cov_func(self.induced_points, self.induced_points)
        L = numpy.linalg.cholesky(self.Ks + self.noise * numpy.eye(
            self.Ks.shape[0]))
        L_inv = solve_triangular(L, numpy.eye(L.shape[0]), lower=True,
                                 check_finite=False)
        self.Ks_inv = L_inv.T.dot(L_inv)
        self.logdet_Ks = 2. * numpy.sum(numpy.log(L.diagonal()))

        self.place_integration_points()
        self.ks_X = self.cov_func(self.induced_points, self.X)
        self.LB_list = []
        self.times = []

        self.kappa_X = self.Ks_inv.dot(self.ks_X)
        self.kappa_int_points = self.Ks_inv.dot(self.ks_int_points)
        self.g_X = self.predictive_g_function(self.X, 'X')
        self.g_int_points = self.predictive_g_function(
            self.integration_points, 'int_points')
        self.alpha0 = 4.
        self.beta0 = 2./(float(self.X.shape[0]/self.R))
        if lmbda_star is None:
            self.lmbda_star = self.alpha0/self.beta0
        else:
            self.lmbda_star = lmbda_star
        self.alpha = self.alpha0
        self.beta = self.beta0
        self.convergence = numpy.inf
        self.conv_crit = conv_crit
        self.num_iterations = 0
        self.output = output
        self.log_likelihood_list = []

    def place_inducing_points(self):
        """ Places the inducing points of the sparse GP.
        """
        num_per_dim = int(numpy.ceil(self.num_inducing_points**(1./self.D)))
        induced_grid = numpy.empty([num_per_dim, self.D])
        for di in range(self.D):
            dist_between_points = self.S[di]/num_per_dim
            induced_grid[:,di] = numpy.arange(.5*dist_between_points,
                                            self.S[di], dist_between_points)

        self.induced_points = numpy.meshgrid(*induced_grid.T.tolist())
        self.induced_points = numpy.array(self.induced_points).reshape([
            self.D, -1]).T

    def place_integration_points(self):
        """ Places the integration points and updates all related kernels.
        """
        self.integration_points = numpy.random.rand(
            self.num_integration_points, self.D)
        self.integration_points *= self.S[numpy.newaxis]
        self.ks_int_points = self.cov_func(self.induced_points,
                                           self.integration_points)
        self.kappa_int_points = self.Ks_inv.dot(self.ks_int_points)

    def cov_func(self, x, x_prime, only_diagonal=False):
        """ Computes the covariance functions between x and x_prime.

        :param x: numpy.ndarray [num_points x D]
            Contains coordinates for points of x
        :param x_prime: numpy.ndarray [num_points_prime x D]
            Contains coordinates for points of x_prime
        :param only_diagonal: bool
            If true only diagonal is computed (Works only if x and x_prime
            are the same, Default=False)

        :return: numpy.ndarray [num_points x num_points_prime]
        ([num_points_prime] if only diagonal)
            Kernel matrix.
        """

        theta_1, theta_2 = self.cov_params[0], self.cov_params[1]
        if only_diagonal:
            return theta_1 * numpy.ones(x.shape[0])

        else:
            x_theta2 = x / theta_2
            xprime_theta2 = x_prime / theta_2
            h = numpy.sum(x_theta2 ** 2, axis=1)[:, None] - 2. * numpy.dot(
                x_theta2, xprime_theta2.T) + \
                numpy.sum(xprime_theta2 ** 2, axis=1)[None]
            return theta_1 * numpy.exp(-.5*h)

    def run(self):
        """ Runs the Expectation-Maximisation algorithm to find the MAP
        estimate.
        """
        self.times.append(time.perf_counter())
        converged = False
        while not converged:
            self.num_iterations += 1
            self.estep()
            self.mstep()
            self.log_likelihood_list.append(self.calculate_log_likelihood())
            if self.num_iterations > 1:
                self.convergence = (self.log_likelihood_list[-1] -
                                    self.log_likelihood_list[
                    -2])/numpy.amax([numpy.abs(self.log_likelihood_list[-1]),
                                     numpy.abs(self.log_likelihood_list[-2]),1])
            converged = self.convergence < self.conv_crit
            self.times.append(time.perf_counter())
            if self.output:
                self.print_info()

    def estep(self):
        """ Updates the posterior of the latent variables.
        """
        self.calculate_PG_expectations()
        self.calculate_posterior_intensity()

    def mstep(self):
        """ Updates the point estimate of GP and max intensity.
        """

        self.calculate_g_s()
        self.update_g_function()
        self.update_max_intensity()

    def calculate_posterior_intensity(self):
        """ The rate of the latent Poisson process is updated.
        """
        self.lmbda = self.lmbda_star/(1. + numpy.exp(self.g_int_points))

    def calculate_PG_expectations(self):
        """ The Polya-Gamma posterior is updated.
        """

        self.c_X = numpy.abs(self.g_X)
        nonzero_idx = numpy.nonzero(self.c_X)
        self.mu_omega_X = numpy.zeros(self.c_X.shape)
        self.mu_omega_X[nonzero_idx] = .5/self.c_X[nonzero_idx]*\
                                       numpy.tanh(.5*self.c_X[nonzero_idx])
        self.c_int_points = numpy.abs(self.g_int_points)
        nonzero_idx = numpy.nonzero(self.c_int_points)
        self.mu_omega_int_points = numpy.zeros(self.c_int_points.shape)
        self.mu_omega_int_points[nonzero_idx] = .5/\
                                                self.c_int_points[nonzero_idx] \
            * numpy.tanh(.5*self.c_int_points[nonzero_idx])

    def update_max_intensity(self):
        """ Updates the the maximal intensity.
        """
        self.alpha = self.X.shape[0] + \
                     numpy.sum(self.lmbda)/self.num_integration_points*self.R +\
                     self.alpha0
        self.beta = self.beta0 + self.R
        self.lmbda_star = (self.alpha - 1.)/self.beta

    def predictive_g_function(self, x_prime, points=None):
        """ Computes the GP at x_prime.

        :param x_prime: numpy.ndarray [num_points x D]
            Points, which should be predicted for.
        :param points: str
            If 'int_points' or 'X' posterior for integration points or
            observation points is calculated, respectively (Default=None).
        :returns:
            numpy.ndarray [num_points]: mean of predictive posterior
            numpy.ndarray [num_points]: variance of predictive posterior
        """
        if points is None:
            ks_x_prime = self.cov_func(self.induced_points, x_prime)
            kappa = self.Ks_inv.dot(ks_x_prime)
        elif points is 'int_points':
            kappa = self.kappa_int_points
        elif points is 'X':
            kappa = self.kappa_X

        g_x_prime = kappa.T.dot(self.g_s)
        return g_x_prime

    def update_g_function(self, only_int_points=False):
        """ Updates the function g (mean & variance) at each point (observed
        and integration points for Monte Carlo integral)

        :param only_int_points: bool
            If True it only updates the integration points (Default=False)
        """

        if not only_int_points:
            self.g_X = self.predictive_g_function(self.X, points='X')
        self.g_int_points = self.predictive_g_function(self.integration_points,
                                                       points='int_points')

    def calculate_g_s(self):
        """ The new GP at the induced points is calculated.
        """

        A_int_points = self.lmbda * self.mu_omega_int_points
        A_X = self.mu_omega_X
        kAk = self.kappa_X.dot(A_X[:,numpy.newaxis] * self.kappa_X.T) + \
                self.kappa_int_points.dot(A_int_points[:,numpy.newaxis] *
                                       self.kappa_int_points.T) \
                / self.num_integration_points * self.R
        W =  kAk + self.Ks_inv
        b_int_points = -.5 * self.lmbda
        b_X = .5 * numpy.ones(self.X.shape[0])
        kb = self.ks_X.dot(b_X) + self.ks_int_points.dot(b_int_points) /\
                             self.num_integration_points * self.R
        self.g_s = numpy.linalg.solve(W,(kb.dot(self.Ks_inv)))

    def calculate_log_likelihood(self):
        """ Calculates the log likelihood function.

        :return: float
            The log-likelihood for training observations
        """
        log_likelihood = numpy.sum(numpy.log(self.lmbda_star/(1. + numpy.exp(
            -self.g_X))))
        log_likelihood -= numpy.mean(
            self.lmbda_star/(1. + numpy.exp(-self.g_int_points)))*self.R
        return log_likelihood

    def create_laplace(self):
        """ Creates the Laplace posterior, by taking the MAP estimate as mean
        and calculating the Hessian of the likelihood, to compute the
        covariance matrix.
        """

        self.mu_g_s = numpy.empty(self.num_inducing_points + 1)
        self.mu_g_s[:-1] = self.g_s
        self.mu_g_s[-1] = numpy.log(self.lmbda_star)

        self.Sigma_g_s_inv = numpy.empty([self.num_inducing_points + 1,
                                          self.num_inducing_points + 1])
        sigmoid_X = 1./(1. + numpy.exp(-self.g_X))
        sigmoid_int_points = 1. / (1. + numpy.exp(-self.g_int_points))

        dsigmoid_X = sigmoid_X * (1. - sigmoid_X)
        dsigmoid_int_points = sigmoid_int_points * (1. - sigmoid_int_points)
        ddsigmoid_int_points = dsigmoid_int_points * \
                               (1. - 2.*sigmoid_int_points)
        kappa_ddsigmoid_int_points = ddsigmoid_int_points*self.kappa_int_points
        self.Sigma_g_s_inv[:-1, :-1] = self.lmbda_star * \
                                       self.kappa_int_points.dot(
             kappa_ddsigmoid_int_points.T) / self.num_integration_points * \
                                       self.R
        kappa_dsigmoid_X = dsigmoid_X * self.kappa_X
        self.Sigma_g_s_inv[:-1, :-1] += self.kappa_X.dot(kappa_dsigmoid_X.T)
        self.Sigma_g_s_inv[:-1, :-1] += self.Ks_inv
        self.Sigma_g_s_inv[-1, -1] = self.lmbda_star * (numpy.mean(
            sigmoid_int_points) * self.R + self.beta0)
        kappa_dsigmoid_int_points = dsigmoid_int_points * self.kappa_int_points
        self.Sigma_g_s_inv[-1, :-1] = self.lmbda_star * numpy.mean(
            kappa_dsigmoid_int_points, axis=1) * self.R
        self.Sigma_g_s_inv[:-1, -1] = self.Sigma_g_s_inv[-1, :-1]
        self.L_inv = numpy.linalg.cholesky(self.Sigma_g_s_inv + self.noise *
                                           numpy.eye(
            self.num_inducing_points + 1))

        self.L = solve_triangular(self.L_inv, numpy.eye(self.L_inv.shape[0]),
                                  lower=True, check_finite=False)
        self.Sigma_g_s = self.L.T.dot(self.L)

    def sample_laplace(self, xprime, num_samples):
        """ Samples Laplace posterior.

        :param xprime: numpy.ndarray [num_points x D]
            Points at which the posterior is sampled.
        :param num_samples: int
            Number of samples that should be drawn from the posterior.

        :return:
            numpy.ndarray [num_points x num_samples]: Samples of GP.
            numpy.ndarray [num_samples]: Samples of max intensity.
        """

        rand_nums = numpy.random.randn(self.num_inducing_points + 1,
                                       num_samples)
        samples = self.mu_g_s[:, None] + numpy.dot(self.L.T,
                                                         rand_nums)
        k = self.cov_func(self.induced_points, xprime)
        mean = k.T.dot(self.Ks_inv.dot(samples[:-1]))
        kprimeprime = self.cov_func(xprime, xprime)
        Sigma = (kprimeprime - k.T.dot(self.Ks_inv.dot(k)))
        L = numpy.linalg.cholesky(Sigma + self.noise * numpy.eye(Sigma.shape[
                                                                     0]))
        g_samples = mean + numpy.dot(L, numpy.random.randn(xprime.shape[0],
                                                        num_samples))
        lmbda_star_samples = numpy.exp(samples[-1])
        return g_samples, lmbda_star_samples

    def predictive_intensity_function(self, X_eval):
        """ Calculates the posterior intensity at X_eval by Gaussian quadrature.

        :param X_eval: numpy.ndarray [num_points x D]
            Points at which the intensity function should be evaluated.
        :return:
            numpy.ndarray [num_points]: Mean of intensity function.
            numpy.ndarray [num_points]: Variance of intensity function.
        """
        num_preds = X_eval.shape[0]
        mean_lmbda_pred, var_lmbda_pred = numpy.empty(num_preds), \
                                          numpy.empty(num_preds)

        mu_rho = self.mu_g_s[-1]
        var_rho = self.Sigma_g_s[-1,-1]
        K_xx = self.cov_func(X_eval, X_eval, only_diagonal=True)
        mu_g_s = self.mu_g_s[:-1]
        Sigma_g_s = self.Sigma_g_s[:-1,:-1]
        cov_gs_rho = self.Sigma_g_s[-1, :-1]
        ks_x_prime = self.cov_func(self.induced_points, X_eval)
        kappa = self.Ks_inv.dot(ks_x_prime)
        tilde_mu_rho_1 = mu_rho + var_rho
        tilde_mu_rho_2 = mu_rho + 2.*var_rho
        tilde_mu_g_s_1 = mu_g_s + cov_gs_rho
        tilde_mu_g_s_2 = mu_g_s + 2.*cov_gs_rho
        factor_1 = numpy.exp(.5 / var_rho * (tilde_mu_rho_1 ** 2 - mu_rho ** 2))
        factor_2 = numpy.exp(.5 / var_rho * (tilde_mu_rho_2 ** 2 - mu_rho ** 2))

        Sigma_g_gs = K_xx - numpy.sum(kappa*ks_x_prime, axis=0)
        Sigma_g = Sigma_g_gs + numpy.sum(kappa * Sigma_g_s.dot(kappa), axis=0)
        for ipred in range(num_preds):

            # mean
            mu1 = numpy.dot(kappa[:,ipred], tilde_mu_g_s_1)
            std = numpy.sqrt(Sigma_g[ipred])
            a, b = mu1 - 10. * std, mu1 + 10. * std
            func1 = lambda g: 1./(1. + numpy.exp(-g)) * \
                             numpy.exp(-.5*((g - mu1)/std)**2) / numpy.sqrt(
                2.*numpy.pi*std**2)
            mean_lmbda_pred[ipred] = factor_1 * quadrature(func1, a, b,
                                                                maxiter=500)[0]
            # var
            mu2 = numpy.dot(kappa[:,ipred], tilde_mu_g_s_2)
            a, b = mu2 - 10. * std, mu2 + 10. * std
            func2 = lambda g: 1. / (1. + numpy.exp(-g))**2 * \
                              numpy.exp(
                                  -.5 * ((g - mu2) / std) ** 2) / numpy.sqrt(
                2. * numpy.pi * std ** 2)

            mean_lmbda_pred_squared = factor_2 * quadrature(func2, a, b,
                                                           maxiter=500)[0]
            var_lmbda_pred[ipred] = mean_lmbda_pred_squared - \
                                    mean_lmbda_pred[ipred]**2

        return mean_lmbda_pred, var_lmbda_pred

    def predictive_log_likelihood(self, X_test, num_samples=1e4):
        """ Samples the predictive test log likelihood.

        :param X_test: numpy.ndarray [num_points x D]
            Points in test set.
        :param num_samples: int
            Number of samples that should be drawn.

        :return: numpy.ndarray [num_samples]
            Predictive log test likelihood for each sample.
        """

        num_events = X_test.shape[0]
        num_samples = int(num_samples)
        X = numpy.concatenate([X_test, self.integration_points])
        num_hundreds = int(num_samples / 1e2)
        pred_log_likelihood = numpy.empty([num_samples])

        # Samples hundred samples at a time.
        for ihundreds in range(num_hundreds):
            g_samples, lmbda_star_samples = self.sample_laplace(X, 100)
            lmbda_sample = lmbda_star_samples / (1. + numpy.exp(-g_samples))

            pred_log_likelihood[ihundreds * 100:(ihundreds + 1) * 100] = \
                numpy.sum(numpy.log(lmbda_sample[:num_events]), axis=0)
            pred_log_likelihood[ihundreds * 100:(ihundreds + 1) * 100] -= \
                numpy.mean(lmbda_sample[num_events:], axis=0) * self.R


        return pred_log_likelihood

