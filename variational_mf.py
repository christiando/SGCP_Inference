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
from scipy.special import digamma, gammaln
from scipy.integrate import quadrature
import time


class VMF_SGCP():

    def __init__(self, S_borders, X, cov_params, num_inducing_points,
                 lmbda_star=None, conv_crit=1e-4,
                 num_integration_points=1000, output=False,
                 update_hyperparams=True,
                 noise=1e-4, epsilon=5e-2):
        """ Class initialisation for variational mean field inference for
        sigmoidal Gaussian Cox process.

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
        :param update_hyperparams: bool
            Whether the hyperparameters are updated (by Adam) or not.  (
            Default=False)
        :param noise: float
            Noise added to the diagonal of the covariance matrix (should be
            small). (Default=1e-4)
        param epsilon: float
            Step size for Adam in the hyperparameter update. (Default=5e-2)
        """

        self.S_borders = S_borders
        self.S = S_borders[:,1] - S_borders[:,0]
        self.R = numpy.prod(self.S)
        self.D = S_borders.shape[0]
        self.noise = noise
        self.cov_params = cov_params
        self.num_integration_points = num_integration_points
        self.num_inducing_points = num_inducing_points  # must be power of D
        self.X = X

        self.place_inducing_points()
        self.mu_g_s = numpy.zeros(self.induced_points.shape[0])
        self.Sigma_g_s = numpy.identity(self.induced_points.shape[0])
        self.Sigma_g_s_inv = numpy.identity(self.induced_points.shape[0])
        self.Ks = self.cov_func(self.induced_points, self.induced_points)
        L = numpy.linalg.cholesky(self.Ks + self.noise * numpy.eye(
            self.Ks.shape[0]))
        L_inv = numpy.linalg.solve(L, numpy.eye(L.shape[0]))
        self.Ks_inv = L_inv.T.dot(L_inv)
        self.logdet_Ks = 2. * numpy.sum(numpy.log(L.diagonal()))

        self.place_integration_points()
        self.ks_X = self.cov_func(self.induced_points, self.X)
        self.LB_list = []
        self.times = []

        self.kappa_X = self.Ks_inv.dot(self.ks_X)
        self.kappa_int_points = self.Ks_inv.dot(self.ks_int_points)
        self.mu_g_X, var_g_X = self.predictive_posterior_GP(self.X, 'X')
        self.mu_g2_X = var_g_X + self.mu_g_X**2
        self.mu_g_int_points, var_g_int_points = self.predictive_posterior_GP(
            self.integration_points, 'int_points')
        self.mu_g2_int_points = var_g_int_points + self.mu_g_int_points**2
        self.epsilon = epsilon
        self.alpha0 = 4.
        self.beta0 = 2./(float(self.X.shape[0]/self.R))
        if lmbda_star is None:
            self.lmbda_star_q1 = self.alpha0/self.beta0
            self.log_lmbda_star_q1 = digamma(self.alpha0)-numpy.log(self.beta0)
        else:
            self.lmbda_star_q1 = lmbda_star
            self.log_lmbda_star_q1 = numpy.log(lmbda_star)
        self.alpha_q1 = self.alpha0
        self.beta_q1 = self.beta0
        self.convergence = numpy.inf
        self.conv_crit = conv_crit
        self.num_iterations = 0
        self.output = output
        self.update_hyperparams = update_hyperparams

        # ADAM parameters
        self.beta1_adam = .9
        self.beta2_adam = .99
        self.epsilon_adam = 1e-5
        self.m_hyper_adam = numpy.zeros(self.D + 1)
        self.v_hyper_adam = numpy.zeros(self.D + 1)
        self.m_bm_adam = numpy.zeros(self.D)
        self.v_bm_adam = numpy.zeros(self.D)

    def place_inducing_points(self):
        """ Places the induced points for sparse GP.
        """

        num_per_dim = int(numpy.ceil(self.num_inducing_points**(1./self.D)))
        induced_grid = numpy.empty([num_per_dim, self.D])
        for di in range(self.D):
            dist_between_points = self.S[di]/num_per_dim
            induced_grid[:,di] = numpy.arange(.5*dist_between_points,
                                            self.S[di],
             dist_between_points)

        self.induced_points = numpy.meshgrid(*induced_grid.T.tolist())
        self.induced_points = numpy.array(self.induced_points).reshape([
            self.D, -1]).T


    def run(self):
        """ Fitting function for the variational mean-field algorithm.
        """

        # Initialisation
        self.times.append(time.perf_counter())
        self.calculate_PG_expectations()
        self.calculate_posterior_intensity()
        converged = False
        while not converged:
            self.num_iterations += 1
            # Update second factor q2
            self.calculate_postrior_GP()
            self.update_predictive_posterior()
            self.update_max_intensity()
            # Update first factor q1
            self.calculate_PG_expectations()
            self.calculate_posterior_intensity()
            # Update hyperparameters
            if self.update_hyperparams:
                self.update_hyperparameters()
            # Calculate lower bound
            self.LB_list.append(self.calculate_lower_bound())
            # Check for convergence
            if self.num_iterations > 1:
                self.convergence = numpy.absolute(self.LB_list[-1] -
                                                 self.LB_list[
                    -2]) / numpy.amax([numpy.abs(self.LB_list[-1]),
                                     numpy.abs(self.LB_list[-2]),1])
            converged = self.convergence < self.conv_crit
            self.times.append(time.perf_counter())
            if self.output:
                self.print_info()

    def print_info(self):
        """ Functions to print info, while iteratively updating posterior.
        """
        print((' +-----------------+ ' +
              '\n |  Iteration %4d |' +
              '\n |  Conv. = %.4f |' +
              '\n +-----------------+') % (self.num_iterations,
                                           self.convergence_inner))

    def place_integration_points(self):
        """ Places the integration points for Monte Carlo integration and
        updates all related kernels.
        """

        self.integration_points = numpy.random.rand(
            self.num_integration_points, self.D)
        self.integration_points *= self.S[numpy.newaxis]
        self.ks_int_points = self.cov_func(self.induced_points,
                                           self.integration_points)
        self.kappa_int_points = self.Ks_inv.dot(self.ks_int_points)

    def calculate_posterior_intensity(self):
        """ The rate of the posterior process is updated.
        """

        self.lmbda_q2 = .5*numpy.exp(
            -.5*self.mu_g_int_points + self.log_lmbda_star_q1)/\
                        numpy.cosh(.5*self.c_int_points)

    def calculate_PG_expectations(self):
        """ The Polya-Gamma posterior is updated.
        """

        self.c_X = numpy.sqrt(self.mu_g2_X)
        self.mu_omega_X = .5/self.c_X*numpy.tanh(
            .5*self.c_X)
        self.c_int_points = numpy.sqrt(self.mu_g2_int_points)
        self.mu_omega_int_points = .5/self.c_int_points \
            * numpy.tanh(.5*self.c_int_points)

    def calculate_predictive_posterior_intensity(self, X_prime):
        """ Calculates the posterior intensity at X_prime for the latent
        Poisson process. (Not the intensity of the observed Poisson process!!!)

        :param X_prime: numpy.ndarray [num_points x D]
            Position of points, that should be evaluated.

        :return: numpy.ndarray [num_points]
            Posterior intensity.
        """
        mu_g, var_g = self.predictive_posterior_GP(X_prime)
        mu_g = mu_g
        mu_g2 = var_g + mu_g ** 2
        c = numpy.sqrt(mu_g2)
        pred_lmbda_q2 = .5 * numpy.exp(
            -.5 * mu_g + self.log_lmbda_star_q1) / \
                        numpy.cosh(.5 * c)
        return pred_lmbda_q2

    def calculate_postrior_GP(self):
        """ The new GP at the inducing points is calculated.
        """

        A_int_points = self.lmbda_q2 * self.mu_omega_int_points
        A_X = self.mu_omega_X
        kAk = self.kappa_X.dot(A_X[:,numpy.newaxis] * self.kappa_X.T) + \
                self.kappa_int_points.dot(A_int_points[:,numpy.newaxis] *
                                       self.kappa_int_points.T) \
                / self.num_integration_points * self.R
        self.Sigma_g_s_inv =  kAk + self.Ks_inv
        L_inv = numpy.linalg.cholesky(self.Sigma_g_s_inv + self.noise *
                                      numpy.eye(
            self.Sigma_g_s_inv.shape[0]))
        L = numpy.linalg.solve(L_inv, numpy.eye(L_inv.shape[0]))
        self.Sigma_g_s = L.T.dot(L)
        self.logdet_Sigma_g_s = 2*numpy.sum(numpy.log(L.diagonal()))
        b_int_points = -.5 * self.lmbda_q2
        b_X = .5 * numpy.ones(self.X.shape[0])
        kb = self.ks_X.dot(b_X) + self.ks_int_points.dot(b_int_points) /\
                             self.num_integration_points * self.R
        self.mu_g_s = self.Sigma_g_s.dot(kb.dot(self.Ks_inv))

    def predictive_posterior_GP(self, x_prime, points=None):
        """ Computes the predictive posterior for given points

        :param x_prime: numpy.ndarray [num_points x D]
            Points, which should be predicted for.
        :param points: str
            If 'int_points' or 'X' posterior for integration points or
            observation points is calculated, respectively. (Default=None)
        :returns:
            numpy.ndarray [num_points]: mean of predictive posterior
            numpy.ndarray [num_points]: variance of predictive posterior
        """
        if points is None:
            ks_x_prime = self.cov_func(self.induced_points, x_prime)
            kappa = self.Ks_inv.dot(ks_x_prime)
        elif points is 'int_points':
            ks_x_prime = self.ks_int_points
            kappa = self.kappa_int_points
        elif points is 'X':
            ks_x_prime = self.ks_X
            kappa = self.kappa_X

        mu_g_x_prime = kappa.T.dot(self.mu_g_s)
        K_xx = self.cov_func(x_prime, x_prime, only_diagonal=True)
        var_g_x_prime = K_xx - numpy.sum(kappa*(ks_x_prime - kappa.T.dot(
            self.Sigma_g_s).T),axis=0)
        return mu_g_x_prime, var_g_x_prime

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

    def calculate_lower_bound(self):
        """ Calculates the variational lower bound for current posterior.

        :return: float
            Variational lower bound.
        """

        Sigma_s_mugmug = self.Sigma_g_s + numpy.outer(self.mu_g_s, self.mu_g_s)
        f_int_points = .5*(- self.mu_g_int_points -
                           self.mu_g2_int_points*self.mu_omega_int_points) -\
                           numpy.log(2)
        integrand = f_int_points - \
                    numpy.log(self.lmbda_q2*numpy.cosh(.5*self.c_int_points)) \
                    + self.log_lmbda_star_q1 + \
                    .5*self.c_int_points**2*self.mu_omega_int_points + 1.
        f_X = .5 * (self.mu_g_X - self.mu_g2_X * self.mu_omega_X) - \
                    numpy.log(2)
        summand = f_X + self.log_lmbda_star_q1 - numpy.log(numpy.cosh(
            .5*self.c_X)) + .5*self.c_X**2*self.mu_omega_X

        L = integrand.dot(self.lmbda_q2)/self.num_integration_points*self.R
        L -= self.lmbda_star_q1*self.R
        L += numpy.sum(summand)
        L -= .5*numpy.trace(self.Ks_inv.dot(Sigma_s_mugmug))
        L -= .5*self.logdet_Ks
        L += .5*self.logdet_Sigma_g_s + .5*self.num_inducing_points
        L += self.alpha0*numpy.log(self.beta0) - gammaln(self.alpha0) + \
             (self.alpha0 - 1)*self.log_lmbda_star_q1 - \
             self.beta0*self.lmbda_star_q1
        L += self.alpha_q1 - numpy.log(self.beta_q1) + gammaln(self.alpha_q1)\
             + (1. - self.alpha_q1)*digamma(self.alpha_q1)

        return L

    def update_max_intensity(self):
        """ Updates the posterior for the maximal intensity.
        """
        self.alpha_q1 = self.X.shape[0] + numpy.sum(
            self.lmbda_q2)/self.num_integration_points*self.R + self.alpha0
        self.beta_q1 = self.beta0 + self.R
        self.lmbda_star_q1 = self.alpha_q1/self.beta_q1
        self.log_lmbda_star_q1 = digamma(self.alpha_q1) - \
                                 numpy.log(self.beta_q1)

    def update_kernels(self):
        """ Updates all kernels (for inducing, observed and integration points).
        """
        self.ks_int_points = self.cov_func(self.induced_points,
                                           self.integration_points)
        self.ks_X = self.cov_func(self.induced_points, self.X)
        self.Ks = self.cov_func(self.induced_points, self.induced_points)
        L = numpy.linalg.cholesky(self.Ks + self.noise * numpy.eye(
            self.Ks.shape[0]))
        L_inv = numpy.linalg.solve(L, numpy.eye(L.shape[0]))
        self.Ks_inv = L_inv.T.dot(L_inv)
        self.logdet_Ks = 2. * numpy.sum(numpy.log(L.diagonal()))
        self.kappa_X = self.Ks_inv.dot(self.ks_X)
        self.kappa_int_points = self.Ks_inv.dot(self.ks_int_points)

    def calculate_hyperparam_derivative(self):
        """ Calculates the derivative of the hyperparameters.

        :return: numpy.ndarray [D + 1]
            Derivatives of hyperparameters.
        """

        theta1, theta2 = self.cov_params[0], numpy.copy(
            self.cov_params[1])
        Sigma_s_mugmug = self.Sigma_g_s + numpy.outer(self.mu_g_s, self.mu_g_s)
        dks_X = numpy.empty([self.ks_X.shape[0], self.ks_X.shape[1],
                             1 + theta2.shape[0]])
        dks_int_points = numpy.empty(
            [self.ks_int_points.shape[0], self.ks_int_points.shape[1],
             1 + theta2.shape[0]])
        dKs = numpy.empty([self.Ks.shape[0], self.Ks.shape[1],
                           1 + theta2.shape[0]])
        dKss = numpy.zeros([1 + theta2.shape[0]])
        dKss[0] = 1.

        # kernel derivatives wrt theta1
        dks_X[:, :, 0] = self.ks_X / theta1
        dks_int_points[:, :, 0] = self.ks_int_points / theta1
        dKs[:, :, 0] = self.Ks / theta1
        # kernel derivatives wrt theta2
        dx = numpy.subtract(self.induced_points[:, None],
                            self.X[None])
        dks_X[:, :, 1:] = self.ks_X[:, :, None] * (dx ** 2) / \
                          (theta2[None, None] ** 3)
        dx = numpy.subtract(self.induced_points[:, None],
                            self.integration_points[None])
        dks_int_points[:, :, 1:] = self.ks_int_points[:, :, None] * \
                                   (dx ** 2) / (theta2[None, None] ** 3)
        dx = numpy.subtract(self.induced_points[:, None],
                            self.induced_points[None])
        dKs[:, :, 1:] = self.Ks[:, :, None] * (dx ** 2) / (
        theta2[None, None] ** 3)
        dL_dtheta = numpy.empty(1 + len(theta2))

        for itheta in range(1 + len(theta2)):
            dKs_inv = -self.Ks_inv.dot(dKs[:, :, itheta].dot(self.Ks_inv))

            dkappa_X = self.Ks_inv.dot(dks_X[:, :, itheta]) + dKs_inv.dot(
                self.ks_X)
            dkappa_int_points = self.Ks_inv.dot(
                dks_int_points[:, :, itheta]) + dKs_inv.dot(
                self.ks_int_points)

            dKtilde_X = dKss[itheta] - numpy.sum(
                dks_X[:, :, itheta] * self.kappa_X, axis=0) - numpy.sum(
                self.ks_X * dkappa_X, axis=0)
            dKtilde_int_points = dKss[itheta] - numpy.sum(
                dks_int_points[:, :, itheta] * self.kappa_int_points,
                axis=0) - numpy.sum(self.ks_int_points * dkappa_int_points,
                                    axis=0)

            dg1_X = self.mu_g_s.dot(dkappa_X)
            dg1_int_points = self.mu_g_s.dot(dkappa_int_points)

            dg2_X = (dKtilde_X + 2. * numpy.sum(
                self.kappa_X * Sigma_s_mugmug.dot(dkappa_X),
                axis=0)) * self.mu_omega_X
            dg2_int_points = (dKtilde_int_points + 2. * numpy.sum(
                self.kappa_int_points * Sigma_s_mugmug.dot(dkappa_int_points),
                axis=0)) * self.mu_omega_int_points

            dL_dtheta[itheta] = .5 * (numpy.sum(dg1_X) - numpy.sum(dg2_X))
            dL_dtheta[itheta] += .5 * numpy.dot(
                -dg1_int_points - dg2_int_points,
                self.lmbda_q2) / self.num_integration_points * self.R
            dL_dtheta[itheta] -= .5 * numpy.trace(self.Ks_inv.dot(
                dKs[:, :, itheta]))
            dL_dtheta[itheta] += .5 * numpy.trace(
                self.Ks_inv.dot(dKs[:, :, itheta].dot(
                    self.Ks_inv.dot(Sigma_s_mugmug))))

        return dL_dtheta


    def update_hyperparameters(self):
        """ Updates the hyperparameters with Adam.
        """
        dL_dtheta = self.calculate_hyperparam_derivative()
        logtheta1, logtheta2 = numpy.log(self.cov_params[0]), \
                               numpy.log(self.cov_params[1])
        dL_dlogtheta1 = dL_dtheta[0] * numpy.exp(logtheta1)
        dL_dlogtheta2 = dL_dtheta[1:] * numpy.exp(logtheta2)

        self.m_hyper_adam[0] = self.beta1_adam * self.m_hyper_adam[0] + \
                               (1. - self.beta1_adam) * dL_dlogtheta1
        self.v_hyper_adam[0] = self.beta2_adam * self.v_hyper_adam[0] + \
                               (1. - self.beta2_adam) * dL_dlogtheta1 ** 2
        self.m_hyper_adam[1:] = self.beta1_adam * self.m_hyper_adam[1:] + \
                                (1. - self.beta1_adam) * dL_dlogtheta2
        self.v_hyper_adam[1:] = self.beta2_adam * self.v_hyper_adam[1:] + \
                                (1. - self.beta2_adam) * dL_dlogtheta2 ** 2
        m_hat = self.m_hyper_adam / (1. - self.beta1_adam)
        v_hat = self.v_hyper_adam / (1. - self.beta2_adam)
        logtheta1 += self.epsilon * m_hat[0] / (numpy.sqrt(v_hat[0]) +
                                                self.epsilon_adam)
        logtheta2 += self.epsilon * m_hat[1:] / (numpy.sqrt(v_hat[1:]) +
                                                 self.epsilon_adam)
        self.cov_params[0] = numpy.exp(logtheta1)
        self.cov_params[1] = numpy.exp(logtheta2)
        self.update_kernels()
        self.update_predictive_posterior()

    def update_predictive_posterior(self, only_int_points=False):
        """ Updates the function g (mean & variance) at each point (observed
        and points for monte carlo integral)

        :param only_int_points: bool
            If True it only updates the integration points. (Default=False)
        """

        if not only_int_points:
            mu_g_X, var_g_X = self.predictive_posterior_GP(
                self.X, points='X')
            self.mu_g_X = mu_g_X
            self.mu_g2_X = var_g_X + mu_g_X ** 2
        mu_g_int_points, var_g_int_points = self.predictive_posterior_GP(
            self.integration_points, points='int_points')
        self.mu_g_int_points = mu_g_int_points
        self.mu_g2_int_points = var_g_int_points + mu_g_int_points ** 2

    def predictive_intensity_function(self, X_eval):
        """ Computes the predictive intensity function at X_eval by Gaussian
        quadrature.

        :param X_eval: numpy.ndarray [num_points_eval x D]
            Points where the intensity function should be evaluated.

        :returns:
            numpy.ndarray [num_points]: mean of predictive posterior intensity
            numpy.ndarray [num_points]: variance of predictive posterior
                                        intensity
        """
        num_preds = X_eval.shape[0]
        mu_pred, var_pred = self.predictive_posterior_GP(X_eval)

        mean_lmbda_pred, var_lmbda_pred = numpy.empty(num_preds), \
                                          numpy.empty(num_preds)

        mean_lmbda_q1 = self.lmbda_star_q1
        var_lmbda_q1 = self.alpha_q1/(self.beta_q1**2)
        mean_lmbda_q1_squared = var_lmbda_q1 + mean_lmbda_q1**2

        for ipred in range(num_preds):
            mu, std = mu_pred[ipred], numpy.sqrt(var_pred[ipred])
            func1 = lambda g_pred: 1. / (1. + numpy.exp(-g_pred)) * \
                                  numpy.exp(-.5*(g_pred - mu)**2 / std**2) / \
                                  numpy.sqrt(2.*numpy.pi*std**2)
            a, b = mu - 10.*std, mu + 10.*std
            mean_lmbda_pred[ipred] = mean_lmbda_q1*quadrature(func1, a, b,
                                                              maxiter=500)[0]
            func2 = lambda g_pred: (1. / (1. + numpy.exp(-g_pred)))**2 * \
                                  numpy.exp(
                                      -.5 * (g_pred - mu) ** 2 / std ** 2) / \
                                  numpy.sqrt(2. * numpy.pi * std ** 2)
            a, b = mu - 10. * std, mu + 10. * std
            mean_lmbda_pred_squared = mean_lmbda_q1_squared *\
                                     quadrature(func2, a, b, maxiter=500)[0]
            var_lmbda_pred[ipred] = mean_lmbda_pred_squared - mean_lmbda_pred[
                                                            ipred]**2

        return mean_lmbda_pred, var_lmbda_pred

    def predictive_log_likelihood(self, X_test, num_samples=1e4):
        """ Samples log predictive likelihood for test set from posterior.

        :param X_test: [num_X_test x D]
            Observations in test set.
        :param num_samples: int
            How many samples of the intensity function should be drawn from
            the posterior. (Default=1e4)

        :return: numpy.ndarray [num_samples]
            Returns the array of sampled likelihoods.
        """

        num_events = X_test.shape[0]
        num_samples = int(num_samples)
        X = numpy.concatenate([X_test, self.integration_points])
        K = self.cov_func(X,X)
        kx = self.cov_func(X,self.induced_points)
        kappa = kx.dot(self.Ks_inv)
        Sigma_post = K - kappa.dot(kx.T - self.Sigma_g_s.dot(kappa.T))
        mu_post = kappa.dot(self.mu_g_s)
        L_post = numpy.linalg.cholesky(Sigma_post + self.noise*numpy.eye(
            Sigma_post.shape[0]))

        num_points = X.shape[0]
        num_hundreds = int(num_samples / 1e2)
        pred_log_likelihood = numpy.empty([num_samples])

        # samples hundred instances at a time
        for ihundreds in range(num_hundreds):
            rand_nums = numpy.random.randn(num_points, 100)
            g_sample = mu_post[:,None] + L_post.dot(rand_nums)
            lmbda_max_sample = numpy.random.gamma(shape=self.alpha_q1,
                                              scale=1./self.beta_q1,
                                              size=100)
            lmbda_sample = lmbda_max_sample / (1. + numpy.exp(-g_sample))
            pred_log_likelihood[ihundreds*100:(ihundreds + 1)*100] = \
                        numpy.sum(numpy.log(lmbda_sample[:num_events]), axis=0)
            pred_log_likelihood[ihundreds*100:(ihundreds + 1)*100] -= \
                numpy.mean(lmbda_sample[num_events:], axis=0)*self.R

        return pred_log_likelihood

    def expanded_predictive_log_likelihood(self, X_test):
        """ Fast approximation for log predictive test likelihood (Eq. 33 in
        paper).

        :param X_test: [num_X_test x D]
            Observations in test set.

        :return: float
            Approximation of log predictive test likelihood.
        """
        self.update_predictive_posterior(only_int_points=True)
        N = X_test.shape[0]
        ks_x_test = self.cov_func(self.induced_points, X_test)
        mu_g_X_test = ks_x_test.T.dot(self.Ks_inv.dot(self.mu_g_s))
        u_mean = -self.lmbda_star_q1*numpy.mean(
            1./(1. + numpy.exp(-self.mu_g_int_points)))*self.R - \
            numpy.sum(numpy.log(1. + numpy.exp(-mu_g_X_test))) + \
            N*numpy.log(self.lmbda_star_q1)

        log_pred_likelihood = u_mean
        du_dg = numpy.empty(N + self.num_integration_points)
        du_dg[:N] = 1./(1. + numpy.exp(mu_g_X_test))
        du_dg[N:] = - self.lmbda_star_q1/(1. + numpy.exp(
            -self.mu_g_int_points)) * (1. - 1./(1. + numpy.exp(
            -self.mu_g_int_points)))\
                    /self.num_integration_points*self.R
        du_dg2 = numpy.empty(N + self.num_integration_points)
        du_dg2[:N] = - (1.- 1./ (1. + numpy.exp(mu_g_X_test))) /\
                     (1. + numpy.exp(mu_g_X_test))
        du_dg2[N:] = - self.lmbda_star_q1 / (1. + numpy.exp(
            -self.mu_g_int_points)) * (1. - 1. / (1. + numpy.exp(
            -self.mu_g_int_points))) * (1. - 2./ (1. + numpy.exp(
            -self.mu_g_int_points))) / self.num_integration_points * self.R

        du_dlambda = - self.R*numpy.mean(
            1./(1. + numpy.exp(-self.mu_g_int_points))) + N/self.lmbda_star_q1
        du_dlmbda2 = - N/self.lmbda_star_q1**2

        C = numpy.empty([N+self.num_integration_points,
                         N+self.num_integration_points])
        inner_matrix = self.Ks_inv.dot(
            numpy.identity(self.num_inducing_points) -
            self.Sigma_g_s.dot(self.Ks_inv))

        K_X = self.cov_func(X_test, X_test) + self.noise*numpy.identity(
            X_test.shape[0])

        C[:N,:N] = K_X - ks_x_test.T.dot(inner_matrix.dot(
            ks_x_test))
        del K_X
        K_int_points = self.cov_func(self.integration_points,
                                     self.integration_points) + \
                       self.noise*numpy.identity(
                           self.integration_points.shape[0])

        C[N:,N:] = K_int_points - self.ks_int_points.T.dot(inner_matrix.dot(
            self.ks_int_points))
        del K_int_points

        K_X_int_points = self.cov_func(self.integration_points, X_test)
        C[N:, :N] = K_X_int_points - self.ks_int_points.T.dot(inner_matrix.dot(
            ks_x_test))
        del K_X_int_points

        C[:N, N:] = C[N:, :N].T

        log_pred_likelihood_corr = .5*numpy.trace(C.dot(numpy.diag(
            du_dg2) + numpy.outer(du_dg, du_dg))) \
            + .5*(du_dlmbda2 + du_dlambda**2)*self.alpha_q1/self.beta_q1**2
        log_pred_likelihood += log_pred_likelihood_corr

        return log_pred_likelihood