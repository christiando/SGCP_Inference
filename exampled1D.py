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

from generative_model import inhomogeneous_poisson_process
from variational_mf import VMF_SGCP
from laplace import Laplace_SGCP
import numpy
from matplotlib import pyplot
from scipy.special import gammaln
import time

num_integration_points = 2000
num_samples = int(2e3)
num_inducing_points = 50
num_test_sets = 10

### Data Generation
num_grid_points = 1000
S_borders = numpy.array([[0, 1]])
lmbda_star = 300.
seed = 1
cov_params = [2., numpy.array([.1])]
X_sets, intensity_sets, X_grid, intensity_array = \
    inhomogeneous_poisson_process(S_borders, lmbda_star, cov_params,
                                  grid_points=num_grid_points, seed=seed,
                                  num_sets=num_test_sets+1)
X = X_sets[0]
X_test = X_sets[1:]
print('Data was generated.')

# Fit variational mean field posterior
var_mf_sgcp = VMF_SGCP(S_borders, X, cov_params, num_inducing_points,
                       num_integration_points=num_integration_points,
                       update_hyperparams=True, output=0, conv_crit=1e-4)
var_mf_sgcp.run()
mu_lmbda_vb, var_lmbda_vb = var_mf_sgcp.predictive_intensity_function(X_grid)
fit_time_vb = var_mf_sgcp.times[-1] - var_mf_sgcp.times[0]
print('Variational posterior was fitted in %.3f s.' %fit_time_vb)

# Gets predictive likelihood
print('Samples test likelihood. Might take a bit.')
num_test_sets = len(X_test)
pred_log_likelihood = numpy.empty([num_test_sets])
pred_log_likelihood_approx = numpy.empty([num_test_sets])
for iset, X_t in enumerate(X_test):
    print('Computing likelihood for test set %d.' %(iset+1))
    pred_log_likelihood_tmp = var_mf_sgcp.predictive_log_likelihood(
        X_t, num_samples=num_samples)
    max_pred_log_likelihood = numpy.amax(pred_log_likelihood_tmp)
    pred_log_likelihood[iset] = numpy.log(numpy.mean(numpy.exp(
        pred_log_likelihood_tmp - max_pred_log_likelihood))) + \
                               max_pred_log_likelihood
    pred_log_likelihood_approx[iset] = \
        var_mf_sgcp.expanded_predictive_log_likelihood(X_t)

mean_pred_log_likelihood = numpy.mean(pred_log_likelihood)
mean_approx_pred_log_likelihood = numpy.mean(pred_log_likelihood_approx)
print('Sampled mean log test likelihood for mean field posterior is %.1f.'
      %mean_pred_log_likelihood)
print('Approx. mean log test likelihood for mean field posterior is %.1f.'
      %mean_approx_pred_log_likelihood)

# Fit Laplace posterior
laplace_sgcp = Laplace_SGCP(S_borders, X, cov_params, num_inducing_points,
                            num_integration_points=num_integration_points,
                            conv_crit=1e-4)
time1_laplace = time.perf_counter()
laplace_sgcp.run()
laplace_sgcp.create_laplace()
fit_time_laplace = time.perf_counter() - time1_laplace
mu_lmbda_laplace, var_lmbda_laplace = \
    laplace_sgcp.predictive_intensity_function(
    X_grid)
print('Laplace posterior was fitted in %.3f s.' %fit_time_laplace)

# Gets test likelihood
pred_log_likelihood = numpy.empty([num_test_sets])
print('Samples test likelihood. Might take a bit.')
for iset, X_t in enumerate(X_test):
    print('Computing likelihood for test set %d.' %(iset+1))
    pred_log_likelihood_tmp = \
        laplace_sgcp.predictive_log_likelihood(X_t, num_samples=num_samples)
    max_pred_log_likelihood = numpy.amax(pred_log_likelihood_tmp)
    pred_log_likelihood[iset] = numpy.log(
        numpy.mean(numpy.exp(pred_log_likelihood_tmp -
                             max_pred_log_likelihood))) + \
                               max_pred_log_likelihood
mean_pred_log_likelihood = numpy.mean(pred_log_likelihood)

print('Sampled mean log test likelihood for Laplace posterior is %.1f.'
      %mean_pred_log_likelihood)

# Plotting
fig = pyplot.figure(figsize=(12,3))
ax_vb = fig.add_subplot(131)
ax_vb.plot(X_grid, intensity_array, color='k')
ax_vb.plot(X_grid, mu_lmbda_vb, color='C0')
ax_vb.fill_between(X_grid[:,0], mu_lmbda_vb - numpy.sqrt(var_lmbda_vb),
                   mu_lmbda_vb + numpy.sqrt(var_lmbda_vb), color='C0',
                   alpha=.7)
ax_vb.vlines(X, 0, lmbda_star*.05,color='k')
ax_vb.set_xlabel('$x$')
ax_vb.set_ylabel('$\Lambda(x)$')

ax_laplace = fig.add_subplot(132)
ax_laplace.plot(X_grid, intensity_array, color='k')
ax_laplace.plot(X_grid, mu_lmbda_laplace, color='C1')
ax_laplace.fill_between(X_grid[:,0],
                        mu_lmbda_laplace - numpy.sqrt(var_lmbda_laplace),
                        mu_lmbda_laplace + numpy.sqrt(var_lmbda_laplace),
                        color='C1', alpha=.7)
ax_laplace.vlines(X, 0, lmbda_star*.05,color='k')
ax_laplace.set_xlabel('$x$')


ax_lambda = fig.add_subplot(133)
lmbda_max_range = numpy.linspace(lmbda_star*.5,lmbda_star*2,200)
dlmbda = lmbda_max_range[1] - lmbda_max_range[0]
alpha_vb, beta_vb = var_mf_sgcp.alpha_q1, var_mf_sgcp.beta_q1
log_lmbda_max_vb = (alpha_vb - 1)*numpy.log(lmbda_max_range) - \
               beta_vb*lmbda_max_range +  alpha_vb*numpy.log(beta_vb) - \
               gammaln(alpha_vb)
lmbda_max_vb = numpy.exp(log_lmbda_max_vb)*dlmbda
mu_laplace, sigma_laplace = laplace_sgcp.mu_g_s[-1], laplace_sgcp.Sigma_g_s[-1, -1]
log_lmbda_max_laplace = - .5*numpy.log(2. * numpy.pi * sigma_laplace) \
                        - .5*(numpy.log(lmbda_max_range) - mu_laplace)**2 \
                        / sigma_laplace - numpy.log(lmbda_max_range)
lmbda_max_laplace = numpy.exp(log_lmbda_max_laplace)*dlmbda

ax_lambda.plot(lmbda_max_range,lmbda_max_vb, color='C0', label='VB')
ax_lambda.plot(lmbda_max_range,lmbda_max_laplace, color='C1', label='Laplace')
max_value = numpy.amax(numpy.concatenate([lmbda_max_vb, lmbda_max_laplace]))
ax_lambda.vlines(lmbda_star,0,max_value, color='k')
ax_lambda.set_xlabel('$\lambda_{\\rm max}$')
ax_lambda.set_ylabel('Posterior of $\lambda_{\\rm max}$')
ax_lambda.legend(frameon=0)

fig.tight_layout()
pyplot.show()