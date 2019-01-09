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

num_integration_points = 2500
num_samples = int(2e3)
num_inducing_points = 100
num_test_sets = 2

### Data Generation
num_grid_points = 60
S_borders = numpy.array([[0, 1],
                         [0, 1]])
lmbda_star = 500.
seed = 1
cov_params = [2., numpy.array([.1, .1])]
X_sets, intensity_sets, X_vec, intensity_array = \
    inhomogeneous_poisson_process(S_borders, lmbda_star, cov_params,
                                  grid_points=num_grid_points, seed=seed,
                                  num_sets=num_test_sets+1)
X_mesh, Y_mesh = numpy.meshgrid(X_vec[:,0],X_vec[:,1])
X_grid = numpy.vstack([X_mesh.flatten(),Y_mesh.flatten()]).T
X = X_sets[0]
X_test = X_sets[1:]
print('Data was generated.')

# Fit variational mean field posterior
var_mf_sgcp = VMF_SGCP(S_borders, X, cov_params, num_inducing_points,
                       num_integration_points=num_integration_points,
                       update_hyperparams=False, output=0, conv_crit=1e-4)
var_mf_sgcp.run()
mu_lmbda_vb, var_lmbda_vb = var_mf_sgcp.predictive_intensity_function(X_grid)
fit_time_vb = var_mf_sgcp.times[-1] - var_mf_sgcp.times[0]
print('Variational posterior was fitted in %.3f s.' %fit_time_vb)

"""
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
"""

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

"""
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
"""

## Plotting
height = 2.
width = 5.
num_plots = 3
height_subplot = .8
factor = height/width
margin_x = .01
margin_y = margin_x/factor

num_per_dim = 10
induced_grid = numpy.empty([num_per_dim, 2])
for di in range(2):
    dist_between_points = 1./num_per_dim
    induced_grid[:,di] = numpy.arange(.5*dist_between_points,
                                    1., dist_between_points)

induced_points = numpy.meshgrid(*induced_grid.T.tolist())
induced_points = numpy.array(induced_points).reshape([2, -1]).T

max_intensity = numpy.amax(numpy.array([numpy.amax(intensity_array),
                             numpy.amax(mu_lmbda_vb),
                             numpy.amax(mu_lmbda_laplace)]))

intensity_true = intensity_array.reshape([num_grid_points, num_grid_points])
mean_intensity_vb = mu_lmbda_vb.reshape([num_grid_points, num_grid_points])
mean_intensity_laplace = mu_lmbda_laplace.reshape([num_grid_points,
                                               num_grid_points])


fig = pyplot.figure(figsize=(width,height))
ax1 = fig.add_axes((margin_x,margin_y,height_subplot*factor,height_subplot))
ax1.pcolor(X_vec[:,0], X_vec[:,1], intensity_true, vmax=max_intensity,
           rasterized=1)
ax1.scatter(X[:,0],X[:,1],s=2, c='C3')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlim([X_vec[0,0],X_vec[-1,0]])
ax1.set_ylim([X_vec[0,1],X_vec[-1,1]])
ax1.set_title('Ground Truth')

ax2 = fig.add_axes((2.*margin_x + height_subplot*factor,margin_y,
                    height_subplot*factor,height_subplot))
ax2.pcolor(X_vec[:,0], X_vec[:,1], mean_intensity_vb, vmax=max_intensity,
           rasterized=1)
ax2.plot(induced_points[:,0],induced_points[:,1],'.',color='C0', ms=2.)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlim([X_vec[0,0],X_vec[-1,0]])
ax2.set_ylim([X_vec[0,1],X_vec[-1,1]])
ax2.set_title('Mean field')

ax3 = fig.add_axes((3.*margin_x + 2*height_subplot*factor,margin_y,
                    height_subplot*factor,height_subplot))
ax3.pcolor(X_vec[:,0], X_vec[:,1], mean_intensity_laplace, vmax=max_intensity,
           rasterized=1)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlim([X_vec[0,0],X_vec[-1,0]])
ax3.set_ylim([X_vec[0,1],X_vec[-1,1]])
ax3.plot(induced_points[:,0],induced_points[:,1],'.',color='C1', ms=2.)
ax3.set_title('Laplace')
pyplot.show()