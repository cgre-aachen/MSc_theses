"""
BaySeg is a Python library for unsupervised clustering of n-dimensional data sets, designed for the segmentation of
one-, two- and three-dimensional data in the field of geological modeling and geophysics. The library is based on the
algorithm developed by Wang et al., 2017 and combines Hidden Markov Random Fields with Gaussian Mixture Models in a
Bayesian inference framework.

************************************************************************************************
References

[1] Wang, H., Wellmann, J. F., Li, Z., Wang, X., & Liang, R. Y. (2017). A Segmentation Approach for Stochastic
    Geological Modeling Using Hidden Markov Random Fields. Mathematical Geosciences, 49(2), 145-177.

************************************************************************************************
@authors: Alexander Schaaf, Hui Wang, Florian Wellmann
************************************************************************************************
BaySeg is licensed under the GNU Lesser General Public License v3.0
************************************************************************************************
"""

import numpy as np  # scientific computing library
from sklearn import mixture  # gaussian mixture model
from scipy.stats import multivariate_normal, norm  # normal distributions
from copy import copy
from itertools import combinations
import tqdm  # smart-ish progress bar
import matplotlib.pyplot as plt  # 2d plotting
from matplotlib import gridspec, rcParams  # plot arrangements
from .colors import cmap, cmap_norm  # custom colormap
from .ie import *
import pandas as pd
from statistics import mode
import heapq as hq

import sys
sys.path.append("C:/Users/Tobias Giesgen/PycharmProjects/gempy")
import gempy as gp

plt.style.use('bmh')  # plot style


class BaySeg:
    def __init__(self, data, n_labels, raw_data, feature_names, boreholes, gp_resolution, beta_init=1, inc_gempy = False,
                 stencil=None, normalize=True, plot=False):
        """

        Args:
            data (:obj:`np.ndarray`): Multidimensional data array containing all observations (features) in the
                following format:

                    1D = (Y, F)
                    2D = (Y, X, F)
                    3D = (Y, X, Z, F)

            n_labels (int): Number of labels representing the number of clusters to be segmented.
            beta_init (float): Initial penalty value for Gibbs energy calculation.
            stencil (int): Number specifying the stencil of the neighborhood system used in the Gibbs energy
                calculation.

        """
        # TODO: [DOCS] Main object description

        # store initial data
        self.data = data
        # get shape for physical and feature dimensions
        self.shape = np.shape(data)
        self.phys_shp = np.array(self.shape[:-1])

        self.inc_gempy = inc_gempy

        # get number of features
        self.n_feat = self.shape[-1]

        # gempy properties
        self.gp_resolution = gp_resolution

        # GRAPH COLORING
        self.stencil = stencil
        self.colors = pseudocolor(self.shape, self.stencil)

        # ************************************************************************************************
        # fetch dimensionality, coordinate and feature vector from input data

        # 1D
        if len(self.shape) == 2:
            # 1d case
            self.dim = 1
            # create coordinate vector
            # self.coords = np.array([np.arange(self.shape[0])]).T
            # feature vector
            self.feat = self.data


        # 2D
        elif len(self.shape) == 3:
            # 2d case
            self.dim = 2
            # create coordinate vector
            # y, x = np.indices(self.shape[:-1])
            # print(y, x)
            # self.coords = np.array([y.flatten(), x.flatten()]).T

            # feature vector
            self.feat = np.array([self.data[:, :, f].ravel() for f in range(self.n_feat)]).T

        # 3D
        elif len(self.shape) == 4:
            # 3d case
            raise Exception("3D segmentation not yet supported.")

        # mismatch
        else:
            raise Exception("Data format appears to be wrong (neither 1-, 2- or 3-D).")

        if normalize:
            self.normalize_feature_vectors()

        # ************************************************************************************************
        # INIT STORAGE ARRAYS

        # self.betas = [beta_init]  # initial beta
        # self.mus = np.array([], dtype=object)
        # self.covs = np.array([], dtype=object)
        # self.labels = np.array([], dtype=object)

        # ************************************************************************************************
        # INIT GAUSSIAN MIXTURE MODEL
        self.n_labels = n_labels
        self.gmm = mixture.GaussianMixture(n_components=n_labels, covariance_type="full")
        self.gmm.fit(self.feat)
        # do initial prediction based on fit and observations, store as first entry in labels

        # ************************************************************************************************
        # INIT LABELS, MU and COV based on GMM
        # TODO: [GENERAL] storage variables from lists to numpy ndarrays
        self.labels = [self.gmm.predict(self.feat)]
        # INIT MU (mean from initial GMM)
        self.mus = [self.gmm.means_]
        # INIT COV (covariances from initial GMM)
        self.covs = [self.gmm.covariances_]

        self.labels_probability = []
        self.storage_gibbs_e = []
        self.storage_like_e = []
        self.storage_te = []
        self.storage_gempy_e = []

        self.beta_acc_ratio = np.array([])
        self.beta_gp_acc_ratio = np.array([])
        self.cov_acc_ratio = np.array([])
        self.mu_acc_ratio = np.array([])

        # ************************************************************************************************
        # Initialize PRIOR distributions for beta, mu and covariance
        # BETA
        if self.dim == 1:
            self.prior_beta = norm(beta_init, np.eye(1) * 100)
            self.betas = [beta_init]
        elif self.dim == 2:
            if self.stencil == "4p":
                beta_dim = 2
            elif self.stencil == "8p" or self.stencil is None:
                beta_dim = 4

            self.betas = [[beta_init for i in range(beta_dim)]]
            self.prior_beta = multivariate_normal([beta_init for i in range(beta_dim)], np.eye(beta_dim) * 100)

        elif self.dim == 3:
            raise Exception("3D not yet supported.")

        if inc_gempy == True:
            if self.stencil == "4p":
                beta_dim2 = 2
            elif self.stencil == "8p" or self.stencil is None:
                beta_dim2 = 4
            self.betas_gp = [[beta_init for i in range(beta_dim2)]]
            self.prior_beta_gp = multivariate_normal([beta_init for i in range(beta_dim2)], np.eye(beta_dim2) * 100)
        else:pass

        # MU
        # generate distribution means for each label
        prior_mu_means = [self.mus[0][label] for label in range(self.n_labels)]
        # generate distribution covariances for each label
        prior_mu_stds = [np.eye(self.n_feat) * 100 for label in range(self.n_labels)]
        # use the above to generate multivariate normal distributions for each label
        self.priors_mu = [multivariate_normal(prior_mu_means[label], prior_mu_stds[label]) for label in
                          range(self.n_labels)]

        # COV
        # generate b_sigma
        self.b_sigma = np.zeros((self.n_labels, self.n_feat))
        for l in range(self.n_labels):
            self.b_sigma[l, :] = np.log(np.sqrt(np.diag(self.gmm.covariances_[l, :, :])))
        # generate kesi
        self.kesi = np.ones((self.n_labels, self.n_feat)) * 100
        # generate nu
        self.nu = self.n_feat + 1
        # *************************************************************************************************
        '''Create Gempy model from initial data'''
        self.boreholes = boreholes
        self.raw_data = raw_data
        self.feature_names = feature_names
        self.n_boreholes = len(boreholes)

    def create_gempy(self, labels, raw_data, labels_prob, plot):
        print('Gempy model under construction:')
        """
        Creates a gempy model with the labels taken from the initial Gaussian Mixture model

        Args:
            labels: labeling (vector containing values corresponding to a label)
            raw_data(pandas_dataframe): X, Y, Z, Well Name, Log I, Log II, Log III...
            feature_names (list with strings): names of the logs (e.g. 'IND' or 'PE')
            boreholes (list of strings): names of the boreholes (e.g. 'BH1' or 'Well1')

        Out: lith_block[0]: label for each grid point in the 3D gempy model (shape: gp_resolution^3 x 1)
                --> reshape(gp_res, gp_res, gp_res)
        """

        self.labels_gp = labels
        self.labels_prob_gp = labels_prob
        # **************************************************************************************************************
        # Zoning each borehole to find the boundaries between the zones or units (based on segmented data)

        self.boundary = []
        self.formation = []
        len_tot = 0

        for i in range(self.n_boreholes):
            self.pos = np.where(self.raw_data == self.boreholes[i])[0]  # split data in boreholes
            self.boundary_temp2, self.formation_temp = self.find_boundaries_new(self.labels_prob_gp[self.pos],
                                                          self.n_labels)  # find boundary in each borehole

            for k in range(self.n_labels):
                self.formation.append(self.formation_temp[k])

            for k in range(self.n_labels - 1):
                # save boundaries ()
                self.boundary.append((self.boundary_temp2[k] + len_tot - 1))
            self.boundary.append(self.pos[-1])
            # print('Borehole', i + 1, 'of', self.n_boreholes, 'is zoned...')

            len_tot = len_tot + len(self.pos)

            self.boundary, self.formation = zip(*sorted(zip(self.boundary, self.formation)))
            self.boundary = list(self.boundary)
            self.formation = list(self.formation)

        print('Borehole zoning finished!')

        # create gempy_file including X,Y,Z,labels,borehole name
        #self.formation = list(range(self.n_boreholes))
        #self.boundary.sort()
        #self.formation.append(mode(self.labels_gp[0:self.boundary[0]]))
        #for k in range(self.n_labels):
            #self.formation.append(mode(self.labels_gp[self.boundary[k]:self.boundary[k+1]]))
        #self.formation = [self.formation for x in range(self.n_labels)]
        #self.formation = [item for sublist in self.formation for item in sublist]

        self.gempy = pd.DataFrame({'X': raw_data.X[self.boundary], 'Y': raw_data.Y[self.boundary],
                                   'Z': raw_data.Z[self.boundary], 'formation': self.formation,
                                   'borehole': raw_data['Well Name'][self.boundary]})

        # rename the layers to put in gempy (x --> layer x+1 | e.g. 0 --> layer 1)
        for k in range(0, 1 + len(set(list(self.gempy['formation'])))):
            self.gempy['formation'] = self.gempy['formation'].replace(to_replace=k, value='Layer %d' % (k + 1))

        # save gempy_input file as csv-file
        self.gempy = self.gempy.sort_values('formation').reset_index(drop=True)

        self.gempy.to_csv('../bayseg/data_temp/GemPy_BaySeg_temp.csv', index=False)

        # execute gempy (load input files | interfaces + orientations)
        self.geo_data = gp.create_data([int(min(self.gempy.X) - min(self.gempy.X)*0.1),
                                        int(max(self.gempy.X) + max(self.gempy.X) *0.1) ,
                                        int(min(self.gempy.Y) - min(self.gempy.Y)*0.1) ,
                                        int(max(self.gempy.Y) + min(self.gempy.Y)*0.1) ,
                                        int(min(self.gempy.Z) + min(self.gempy.Z)*0.1) , 0],
                                       [self.gp_resolution, self.gp_resolution, self.gp_resolution],
                                       # path_o="../data/Gempy_Simple_4_layer_90degrees_orientation.csv",
                                       path_i="../bayseg/data_temp/GemPy_BaySeg_temp.csv")

        # extract the stratigraphic order from the saved formations
        self.series = pd.unique(np.flipud(self.geo_data.interfaces.sort_values('Z').formation)[:-1])
            #list(self.gempy.loc[np.where(self.gempy.borehole == self.boreholes[0])[0]]
                          # .sort_values('Z',ascending = False).formation.values)

        # set orientation information
        self.series = pd.unique(np.flipud(self.geo_data.interfaces.sort_values('Z').formation)[:-1])
        if 'basement' in self.series:
            self.series = np.delete(self.series, np.where(self.series == 'basement')[0][0])
            #list(self.gempy.loc[np.where(self.gempy.borehole == self.boreholes[0])[0]]
                          # .sort_values('Z',ascending = False).formation.values)

        # set orientation information
        self.save_ori = []
        for i in range(1,self.n_labels+1):
            ori_points = self.nearest_points(i)
            gp.set_orientation_from_interfaces(self.geo_data, ori_points)
            self.save_ori.append(ori_points)

        # set stratigraphic pile for gempy
        gp.set_series(self.geo_data, {"Strat_Series": list(self.series)},
                      order_series=["Strat_Series"],
                      order_formations=list(self.series), verbose=0)

        # interpolate the data in the 3D gempy model (full theano compiling only for first iteration)
        if len(self.labels) == 1:
            self.interp_data = gp.InterpolatorData(self.geo_data, u_grade=[1], output='geology', compile_theano=True,
                                                   dtype='float64', theano_optimizer='fast_compile')
        else: self.interp_data.update_interpolator(self.geo_data)

        # compute gempy model
        self.lith_block, self.fault_block = gp.compute_model(self.interp_data)

        # activate plotting of gempy model section
        if len(self.labels) == self.n_inter:
            if plot == '2dx':
                gp.plotting.plot_section(self.geo_data, self.lith_block[0], cell_number=5, direction='x', plot_data=True)
                # gp.plot_section(self.geo_data, self.lith_block[0], cell_number=5, direction='x', plot_data=True)
            elif plot == '2dy':
                gp.plotting.plot_section(self.geo_data, self.lith_block[0], cell_number=5, direction='y', plot_data=True)
                # gp.plot_section(self.geo_data, self.lith_block[0], cell_number=5, direction='y', plot_data=False)
            elif plot == '3d':
                ver_s, sim_s = gp.get_surfaces(self.interp_data, self.lith_block[1], original_scale=True)
                gp.plotting.plot_surfaces_3D_real_time(self.geo_data,self.interp_data, ver_s, sim_s)
            else: pass

        # renaming lith_block in the same way as self.labels (gempy renames the labels from 1 to n_labels)
        self.rename_lith_block()
        #self.lith_block[0] = np.round(self.lith_block[0], 0)

        #for i in range(self.n_labels):
            #np.place(self.lith_block[0], self.lith_block[0] == i + 1,
                     #[int(s) for s in self.series[i].split() if s.isdigit()][0] - 1 + 10)
        #self.lith_block[0] = (self.lith_block[0] - 10)  # +10 -10 to avoid overwriting in loop

        # create index_gempy (only in first iteration because do not change)
        self.gempy_indicies()

        print('Gempy model finished!')

    def rename_lith_block(self):
        self.lith_block[0] = np.round(self.lith_block[0], 0)

        name_temp = []
        name_temp.append(self.labels[-1][0:self.boundary[0]])

        for i in range(self.n_labels - 1):
            name_temp.append(self.labels[-1][self.boundary[i]:self.boundary[i + 1]])

        for k in range(self.n_labels):
            lab = np.where(np.bincount(name_temp[k], minlength=self.n_labels) ==
                           max(np.bincount(name_temp[k], minlength=self.n_labels)))[0][0]

            self.lith_block[0][np.where(self.lith_block[0] == k + 1)[0]] = lab + 10

        self.lith_block[0] = self.lith_block[0] - 10

    def gempy_indicies(self):
        """
            Finds the indicies in the 3D gempy grid, which are the closest to the borehole data points

        Out: positions of the boundaries in the borehole
        """
        # build vector containing the coordinates of all borehole data points
        self.coords = self.raw_data.loc[:,['X','Y','Z']].values

        self.index_temp = []
        self.index_gempy = []

        # reshape the grid in the same way as the gempy labels for the gibbs energy calculation
        for i in range(self.gp_resolution):
            self.index_temp.append(self.geo_data.grid.values.reshape(
                self.gp_resolution, self.gp_resolution,
                self.gp_resolution, 3)[:, :, i, :].reshape(
                self.gp_resolution ** 2, 3))

        self.index_temp = np.asarray(self.index_temp).reshape(self.gp_resolution ** 3, 3)

        # index_gempy =  minimum (gempy_coords - borehole_coords[i])
        for i in range(len(self.coords)):
            self.index_gempy.append(np.sum(np.abs(self.index_temp - self.coords[i]), axis=1).argmin())

    def find_boundaries_new(self, labels_proba, n):
        max_prob = np.sum(labels_proba, axis=0)

        likelihood_sum = []
        for i in range(len(labels_proba) + 1):
            likelihood_sum.append(np.sum(labels_proba[0:i], axis=0))

        likelihood_sum_norm = likelihood_sum/max_prob

        b_temp = []
        formation = []
        for k in range(n):
            lst = np.arange(n)
            lst = np.delete(lst, k)
            cum_like = n*likelihood_sum_norm[:,k] - np.sum(likelihood_sum_norm[:,lst],axis=1)
            b_temp.append(np.where(cum_like == max(cum_like))[0][0])
            formation.append(k)

        b_temp.sort()
        b_temp = b_temp[0:n-1]
        return b_temp, formation

    def find_boundaries(self, seg_data, n):
        """
            Finds the zone boundaries, which minimize the variance in each zones
                --> applied to each borehole separately

        Args:
            seg_data: labels for each data point in one borehole
            n: number of zones

        Out: positions of the boundaries in the borehole
        """

        var_t = []
        var_bz = []

        for i in range(2, len(seg_data) - (n - 1)):
            if n == 2:
                var_t.append((np.sum(np.var(seg_data[0:i])) + np.sum(np.var(seg_data[i:-1])), i))
                var_bz.append((np.var([np.mean(seg_data[0:i]), np.mean(seg_data[i:-1])]), i))
            else:
                for j in range(i + 2, len(seg_data) - (n - 2)):
                    if n == 3:
                        var_t.append((np.sum(np.var(seg_data[0:i])) + np.sum(np.var(seg_data[i:j])) + np.sum(
                            np.var(seg_data[j:-1])), i, j))
                        var_bz.append(
                            (np.var([np.mean(seg_data[0:i]), np.mean(seg_data[i:j]), np.mean(seg_data[j:-1])]), i, j))
                        # boundaries = np.where(var_t == min(var_t))[0][0] + 2
                    else:
                        for k in range(j + 2, len(seg_data) - (n - 3)):
                            if n == 4:
                                var_t.append((np.sum(np.var(seg_data[0:i])) + np.sum(np.var(seg_data[i:j])) + np.sum(
                                    np.var(seg_data[j:k])) + np.sum(np.var(seg_data[k:-1])), i, j, k))
                                var_bz.append((np.var(
                                    [np.mean(seg_data[0:i]), np.mean(seg_data[i:j]), np.mean(seg_data[j:k]),
                                     np.mean(seg_data[k:-1])]), i, j, k))
                            else:
                                for l in range(k + 2, len(seg_data) - (n - 4)):  # check from here
                                    if n == 5:
                                        var_t.append((
                                            np.sum(np.var(seg_data[0:i])) + np.sum(np.var(seg_data[i:j])) + np.sum(
                                                np.var(seg_data[j:k])) + np.sum(np.var(seg_data[k:l])) + np.sum(
                                                np.var(seg_data[l:-1])), i, j, k, l))
                                        var_bz.append((np.var(
                                            [np.mean(seg_data[0:i]), np.mean(seg_data[i:j]), np.mean(seg_data[j:k]),
                                             np.mean(seg_data[k:-1])]), i, j, k))
                                    else:
                                        for m in range(l + 2, len(seg_data) - (n - 5)):
                                            if n == 6:
                                                var_t.append((np.sum(np.var(seg_data[0:i])) + np.sum(
                                                    np.var(seg_data[i:j])) + np.sum(np.var(seg_data[j:k])) + + np.sum(
                                                    np.var(seg_data[k:l])) + np.sum(np.var(seg_data[l:m])) + np.sum(
                                                    np.var(seg_data[m:-1])), i, j, k, l, m))
                                                var_bz.append((np.var(
                                                    [np.mean(seg_data[0:i]), np.mean(seg_data[i:j]),
                                                     np.mean(seg_data[j:k]),
                                                     np.mean(seg_data[k:-1])]), i, j, k))
                                            else:
                                                pass

        return min(var_t)[1:n]  # , max(var_bz)[1:n]] # for maximum variance between clusters

    def calc_gempy_energy_vect(self, beta, verbose=False):
        # Calculate gempy energy (gibbs energy from surroundings
        # need to be done plain wise bbecause of 2D --> gempy.plains (plains with constant z)
        # insert only one plain as vector (Y x 1)
        # print(np.round(self.lith_block[]))

        self.gempy_plains = self.lith_block[0].reshape(self.gp_resolution, self.gp_resolution, self.gp_resolution)
        self.gempy_energy_temp = []

        # calculate gibbs energy for each gempy Z-plain
        for i in range(self.gp_resolution):
            self.gempy_energy_temp.append(
                    self._calc_gibbs_energy_vect(self.gempy_plains[:, :, i].reshape(self.gp_resolution ** 2),
                        beta, dim=2, verbose=verbose))  # 2D energy from Gempy mode
        # reshape
        self.gempy_energy_temp_total = np.asarray(self.gempy_energy_temp).reshape(self.gp_resolution ** 3, self.n_labels)
        return self.gempy_energy_temp_total[self.index_gempy]

    # ************************************************************************************************
    def fit(self, n, beta_jump_length=10, mu_jump_length=0.0005, cov_volume_jump_length=0.00005,
            theta_jump_length=0.0005, t=1., verbose=False, fix_beta=False, plot=False):
        """Fit the segmentation parameters to the given data.

        Args:
            n (int): Number of iterations.
            beta_jump_length (float): Hyperparameter specifying the beta proposal jump length.
            mu_jump_length (float): Hyperparameter for the mean proposal jump length.
            cov_volume_jump_length (float):
            theta_jump_length (float):
            t (float):
            verbose (bool or :obj:`str`):
            fix_beta (bool):

        """
        self.n_inter = n
        for g in tqdm.trange(n):
            self.gibbs_sample(t, beta_jump_length, mu_jump_length, cov_volume_jump_length, theta_jump_length,
                              verbose, fix_beta, plot)

    def gibbs_sample(self, t, beta_jump_length, mu_jump_length, cov_volume_jump_length, theta_jump_length, verbose,
                     fix_beta, plot):
        """Takes care of the Gibbs sampling. This is the main function of the algorithm.

        Args:
            t: Hyperparameter
            beta_jump_length: Hyperparameter
            mu_jump_length: Hyperparameter
            cov_volume_jump_length: Hyperparameter
            theta_jump_length: Hyperparameter
            verbose (bool or :obj:`str`): Toggles verbosity.
            fix_beta (bool): Fixed beta to the inital value if True, else adaptive.

        Returns:
            The function updates directly on the object variables and appends new draws of labels and
            parameters to their respective storages.
        """
        # TODO: [GENERAL] In-depth description of the gibbs sampling function

        # ************************************************
        # CALCULATE TOTAL ENERGY
        # 1 - calculate energy likelihood for each element and label
        # way to avoid over-smoothing by the gibbs energy
        energy_like = self.calc_energy_like(self.mus[-1], self.covs[-1])
        if verbose == "energy":
            print("likelihood energy:", energy_like)
        # 2 - calculate gibbs/mrf energy
        gibbs_energy = self._calc_gibbs_energy_vect(self.labels[-1], self.betas[-1], dim=1, verbose=verbose)
        if verbose == "energy":
            print("gibbs energy:", gibbs_energy)

        # 5 - calculate total energy without gempy model
        total_energy = energy_like + gibbs_energy
        # CALCULATE PROBABILITY OF LABELS  without gempy model
        labels_prob = _calc_labels_prob(total_energy, t)

        if self.inc_gempy == True and len(self.labels) == 1:
            # create gempy model with old labels and calculate gempy energy
            self.create_gempy(self.labels[-1], self.raw_data, labels_prob, plot)
            gempy_energy = self.calc_gempy_energy_vect(self.betas_gp[-1])
            # 5 - calculate total energy with gempy model
            total_energy = energy_like + gibbs_energy + gempy_energy
            # CALCULATE PROBABILITY OF LABELS with gempy model
            labels_prob = _calc_labels_prob(total_energy, t)
        else: pass

        self.storage_te.append(total_energy)

        # make copy of previous labels
        new_labels = copy(self.labels[-1])

        # draw new random sample and update old labeling
        # TODO: create GemPy model and GemPy energy in each iteration of updating labels
        # TODO: just insert the GemPy Energy based on old GemPy model?
        # TODO: can be neglected because of computational time
        # TODO: GemPy model is not considered when updating labels so far
        for i, color_f in enumerate(self.colors):
            new_labels[color_f] = draw_labels_vect(labels_prob[color_f])
            # now recalculate gibbs energy and other energies from the mixture of old and new labels
            gibbs_energy = self._calc_gibbs_energy_vect(new_labels, self.betas[-1], dim=1, verbose=verbose)
            total_energy = energy_like + gibbs_energy
            labels_prob = _calc_labels_prob(total_energy, t)

        if self.inc_gempy == True:
            # create gempy model with old labels and calculate gempy energy
            self.create_gempy(new_labels, self.raw_data, labels_prob, plot)
            gempy_energy = self.calc_gempy_energy_vect(self.betas_gp[-1])
            total_energy = energy_like + gibbs_energy + gempy_energy
            labels_prob = _calc_labels_prob(total_energy, t)
        else: pass

        self.labels_probability.append(labels_prob)
        self.labels.append(new_labels)

        # ************************************************************************************************
        # calculate energy for component coefficient
        # TODO: Check what component coefficient is and maybe add gempy energy
        if self.inc_gempy == True:
            energy_for_comp_coef = gibbs_energy
            energy_for_comp_coef_gp = gempy_energy
        else: energy_for_comp_coef = gibbs_energy
        # print("ge shp:", gibbs_energy)
        # ************************************************************************************************
        # CALCULATE COMPONENT COEFFICIENT
        comp_coef = _calc_labels_prob(energy_for_comp_coef, t)
        if self.inc_gempy == True:
            comp_coef_gp = _calc_labels_prob(energy_for_comp_coef_gp, t)
        # ************************************************************************************************
        # ************************************************************************************************
        # PROPOSAL STEP
        # make proposals for beta, mu and cov
        # beta depends on physical dimensions, for 1d its size 1
        beta_prop = self.propose_beta(self.betas[-1], beta_jump_length, dim = 1)

        if self.inc_gempy == True:
            beta_prop_gp = self.propose_beta(self.betas_gp[-1], beta_jump_length, dim = 2)
        else: pass

        # print("beta prop:", beta_prop)
        mu_prop = self.propose_mu(self.mus[-1], mu_jump_length)
        # print("mu prop:", mu_prop)
        cov_prop = _propose_cov(self.covs[-1], self.n_feat, self.n_labels, cov_volume_jump_length, theta_jump_length)
        # print("cov_prop:", cov_prop)

        # ************************************************************************************************
        # Compare mu, cov and beta proposals with previous, then decide which to keep for next iteration

        # prepare next ones
        mu_next = copy(self.mus[-1])
        cov_next = copy(self.covs[-1])

        # ************************************************************************************************
        # UPDATE MU
        for l in range(self.n_labels):
            # log-prob prior density for mu
            mu_temp = copy(mu_next)
            mu_temp[l, :] = mu_prop[l, :]

            lp_mu_prev = self.log_prior_density_mu(mu_next, l)
            lp_mu_prop = self.log_prior_density_mu(mu_temp, l)

            lmd_prev = self.calc_sum_log_mixture_density(comp_coef, mu_next, cov_next)
            # calculate log mixture density for proposed mu and cov
            lmd_prop = self.calc_sum_log_mixture_density(comp_coef, mu_temp, cov_next)

            # combine
            log_target_prev = lmd_prev + lp_mu_prev
            log_target_prop = lmd_prop + lp_mu_prop

            mu_eval = evaluate(log_target_prop, log_target_prev)
            if mu_eval[0]:
                mu_next[l, :] = mu_prop[l, :]
            else:
                pass
            self.mu_acc_ratio = np.append(self.mu_acc_ratio, mu_eval[1])

        self.mus.append(mu_next)

        # ************************************************************************************************
        # UPDATE COVARIANCE
        for l in range(self.n_labels):
            cov_temp = copy(cov_next)
            cov_temp[l, :, :] = cov_prop[l, :, :]

            # print("cov diff:", cov_next[l, :, :]-cov_temp[l, :, :])

            # log-prob prior density for covariance
            lp_cov_prev = self.log_prior_density_cov(cov_next, l)
            # print("lp_cov_prev:", lp_cov_prev)
            lp_cov_prop = self.log_prior_density_cov(cov_temp, l)
            # print("lp_cov_prop:", lp_cov_prop)

            lmd_prev = self.calc_sum_log_mixture_density(comp_coef, mu_next, cov_next)
            # print("lmd_prev:", lmd_prev)
            # calculate log mixture density for proposed mu and cov
            lmd_prop = self.calc_sum_log_mixture_density(comp_coef, mu_next, cov_temp)
            # print("lmd_prop:", lmd_prop)

            # combine
            log_target_prev = lmd_prev + lp_cov_prev
            log_target_prop = lmd_prop + lp_cov_prop

            mu_eval = evaluate(log_target_prop, log_target_prev)
            if mu_eval[0]:
                cov_next[l, :] = cov_prop[l, :]
            else:
                pass
            self.cov_acc_ratio = np.append(self.cov_acc_ratio, mu_eval[1])

        # append cov and mu
        self.covs.append(cov_next)
        self.storage_gibbs_e.append(gibbs_energy)
        self.storage_like_e.append(energy_like)

        if self.inc_gempy == True:
            self.storage_gempy_e.append(gempy_energy)
        else: pass

        if not fix_beta:
            # ************************************************************************************************
            # UPDATE BETA FOR Well data
            lp_beta_prev = self.log_prior_density_beta(self.betas[-1])
            lp_beta_prop = self.log_prior_density_beta(beta_prop)

            lmd_prev = self.calc_sum_log_mixture_density(comp_coef, self.mus[-1], self.covs[-1])

            # calculate gibbs energy with new labels and proposed beta
            gibbs_energy_prop = self._calc_gibbs_energy_vect(self.labels[-1], beta_prop, dim=1, verbose=verbose)
            energy_for_comp_coef_prop = gibbs_energy_prop  # + self_energy
            comp_coef_prop = _calc_labels_prob(energy_for_comp_coef_prop, t)

            lmd_prop = self.calc_sum_log_mixture_density(comp_coef_prop, self.mus[-1], self.covs[-1])
            # print("lmd_prev:", lmd_prev)
            # print("lp_beta_prev:", lp_beta_prev)
            log_target_prev = lmd_prev + lp_beta_prev
            # print("lmd_prop:", lmd_prop)
            # print("lp_beta_prop:", lp_beta_prop)
            log_target_prop = lmd_prop + lp_beta_prop

            mu_eval = evaluate(log_target_prop, log_target_prev)
            if mu_eval[0]:
                self.betas.append(beta_prop)
            else:
                self.betas.append(self.betas[-1])
            self.beta_acc_ratio = np.append(self.beta_acc_ratio, mu_eval[1])  # store

            # UPDATE BETA FOR GemPY model
            if self.inc_gempy == True:
                lp_beta_prev = self.log_prior_density_beta(self.betas_gp[-1], dim =2)
                lp_beta_prop = self.log_prior_density_beta(beta_prop_gp, dim = 2)

                lmd_prev = self.calc_sum_log_mixture_density(comp_coef_gp, self.mus[-1], self.covs[-1])

                # calculate gibbs energy with new labels and proposed beta
                gempy_energy_prop = self.calc_gempy_energy_vect(beta_prop_gp)
                energy_for_comp_coef_prop_gp = gempy_energy_prop  # + self_energy
                comp_coef_prop_gp = _calc_labels_prob(energy_for_comp_coef_prop_gp, t)

                lmd_prop = self.calc_sum_log_mixture_density(comp_coef_prop_gp, self.mus[-1], self.covs[-1])
                # print("lmd_prev:", lmd_prev)
                # print("lp_beta_prev:", lp_beta_prev)
                log_target_prev = lmd_prev + lp_beta_prev
                # print("lmd_prop:", lmd_prop)
                # print("lp_beta_prop:", lp_beta_prop)
                log_target_prop = lmd_prop + lp_beta_prop

                mu_eval = evaluate(log_target_prop, log_target_prev)
                if mu_eval[0]:
                    self.betas_gp.append(beta_prop_gp)
                else:
                    self.betas_gp.append(self.betas_gp[-1])
                self.beta_acc_ratio = np.append(self.beta_gp_acc_ratio, mu_eval[1])  # store


        else:
            self.betas.append(self.betas[-1])
            self.betas_gp.append(self.betas_gp[-1])
            # ************************************************************************************************
        # **********************************************************

    def log_prior_density_mu(self, mu, label):
        """Calculates the summed log prior density for a given mean and labels array."""
        with np.errstate(divide='ignore'):
            return np.sum(np.log(self.priors_mu[label].pdf(mu)))

    def log_prior_density_beta(self, beta, dim = 1):
        """Calculates the log prior density for a given beta array."""
        if dim == 1:
            return np.log(self.prior_beta.pdf(beta))
        else:
            return np.log(self.prior_beta_gp.pdf(beta))

    def log_prior_density_cov(self, cov, l):
        """Calculates the summed log prior density for the given covariance matrix and labels array."""
        lam = np.sqrt(np.diag(cov[l, :, :]))
        r = np.diag(1. / lam) @ cov[l, :, :] @ np.diag(1. / lam)
        logp_r = -0.5 * (self.nu + self.n_feat + 1) * np.log(np.linalg.det(r)) - self.nu / 2. * np.sum(
            np.log(np.diag(np.linalg.inv(r))))
        logp_lam = np.sum(np.log(multivariate_normal(mean=self.b_sigma[l, :], cov=self.kesi[l, :]).pdf(np.log(lam.T))))
        return logp_r + logp_lam

    def propose_beta(self, beta_prev, beta_jump_length, dim = 1):
        """Proposes a perturbed beta based on a jump length hyperparameter.

        Args:
            beta_prev:
            beta_jump_length:

        Returns:

        """
        # create proposal covariance depending on physical dimensionality
        # dim = [1, 4, 13]
        if dim == 1:
            beta_dim = 1

        elif dim == 2:
            if self.stencil == "4p":
                beta_dim = 2
            elif self.stencil == "8p" or self.stencil is None:
                beta_dim = 4

        elif dim == 3:
            raise Exception("3D not yet supported.")

        sigma_prop = np.eye(beta_dim) * beta_jump_length
        # draw from multivariate normal distribution and return
        # return np.exp(multivariate_normal(mean=np.log(beta_prev), cov=sigma_prop).rvs())
        return multivariate_normal(mean=beta_prev, cov=sigma_prop).rvs()

    def propose_mu(self, mu_prev, mu_jump_length):
        """Proposes a perturbed mu matrix using a jump length hyperparameter.

        Args:
            mu_prev (:obj:`np.ndarray`): Previous mean array for all labels and features
            mu_jump_length (float or int): Hyperparameter specifying the jump length for the new proposal mean array.

        Returns:
            :obj:`np.ndarray`: The newly proposed mean array.

        """
        # prepare matrix
        mu_prop = np.ones((self.n_labels, self.n_feat))
        # loop over labels
        for l in range(self.n_labels):
            mu_prop[l, :] = multivariate_normal(mean=mu_prev[l, :], cov=np.eye(self.n_feat) * mu_jump_length).rvs()
        return mu_prop

    def calc_sum_log_mixture_density(self, comp_coef, mu, cov):
        """Calculate sum of log mixture density with each observation at every element.

        Args:
            comp_coef (:obj:`np.ndarray`): Component coefficient for each element (row) and label (column).
            mu (:obj:`np.ndarray`): Mean value array for all labels and features.
            cov (:obj:`np.ndarray`): Covariance matrix.

        Returns:
            float: Summed log mixture density.

        """
        lmd = np.zeros((self.phys_shp.prod(), self.n_labels))

        for l in range(self.n_labels):
            draw = multivariate_normal(mean=mu[l, :], cov=cov[l, :, :]).pdf(self.feat)
            # print(np.shape(lmd[:,l]))
            multi = comp_coef[:, l] * np.array([draw])
            lmd[:, l] = multi
        lmd = np.sum(lmd, axis=1)
        with np.errstate(divide='ignore'):
            lmd = np.log(lmd)

        return np.sum(lmd)

    def calc_energy_like(self, mu, cov):
        """Calculates the energy likelihood for a given mean array and covariance matrix for the entire domain.

        Args:
            mu (:obj:`np.ndarray`):
            cov (:obj:`np.ndarray`):
            vect (bool, optional): Toggles the vectorized implementation. False activates the loop-based version if
                you really dig a loss of speed of about 350 times.

        Returns:
            :obj:`np.ndarray` : Energy likelihood for each label at each element.
        """
        energy_like_labels = np.zeros((self.phys_shp.prod(), self.n_labels))

        # uses flattened features array
        for l in range(self.n_labels):
            energy_like_labels[:, l] = np.einsum("...i,ji,...j",
                                                 0.5 * np.array([self.feat.values - mu[l, :]]),
                                                 np.linalg.inv(cov[l, :, :]),
                                                 np.array([self.feat.values - mu[l, :]])) + 0.5 * np.log(
                np.linalg.det(cov[l, :, :]))

        return energy_like_labels

    def _calc_gibbs_energy_vect(self, labels, beta, dim, verbose=False):
        """Calculates the Gibbs energy for each element using the penalty factor(s) beta.

        Args:
            labels (:obj:`np.ndarray`):
            beta (:obj:`np.array` of float):
            verbose (bool):

        Returns:
            :obj:`np.ndarray` : Gibbs energy at every element for each label.
        """
        # ************************************************************************************************
        # 1D

        if dim == 1:
            # tile
            lt = np.tile(labels, (self.n_labels, 1)).T

            ge = np.arange(self.n_labels)  # elements x labels
            ge = np.tile(ge, (len(labels), 1)).astype(float)

            # first row
            top = [np.not_equal(np.arange(self.n_labels), lt[1, :]) * beta]
            # mid
            mid = (np.not_equal(ge[1:-1, :], lt[:-2, :]).astype(float) + np.not_equal(ge[1:-1, :], lt[2:, :]).astype(
                float)) * beta
            # last row
            bot = [np.not_equal(np.arange(self.n_labels), lt[-2, :]) * beta]
            # put back together and return gibbs energy
            return np.concatenate((top, mid, bot))

        # ************************************************************************************************
        # 2D
        elif dim == 2:

            # TODO: Reshape according points that are insert
            # reshape the labels to 2D for "stencil-application"
            labels = labels.reshape(self.gp_resolution, self.gp_resolution)  # self.shape[0], self.shape[1])

            # prepare gibbs energy array (filled with zeros)
            ge = np.tile(np.zeros_like(labels).astype(float), (self.n_labels, 1, 1))

            # create comparison array containing the different labels
            comp = np.tile(np.zeros_like(labels), (self.n_labels, 1, 1)).astype(float)
            for i in range(self.n_labels):
                comp[i, :, :] = i

            # anisotropic beta directions
            #  3  1  2
            #   \ | /
            #   --+-- 0
            #   / | \

            # **********************************************************************************************************
            # direction 0 = 0° polar coord system
            ge[:, 1:-1, 1:-1] += (np.not_equal(comp[:, 1:-1, 1:-1], labels[:-2, 1:-1]).astype(
                float)  # compare with left
                                  + np.not_equal(comp[:, 1:-1, 1:-1], labels[2:, 1:-1]).astype(float)) * beta[0]
            # compare with right

            # left column
            # right
            ge[:, :, 0] += np.not_equal(comp[:, :, 0], labels[:, 1]).astype(float) * beta[0]
            # right column
            # left
            ge[:, :, -1] += np.not_equal(comp[:, :, -1], labels[:, -2]).astype(float) * beta[0]
            # top row
            # right
            ge[:, 0, :-1] += np.not_equal(comp[:, 0, :-1], labels[0, 1:]).astype(float) * beta[0]
            # left
            ge[:, 0, 1:] += np.not_equal(comp[:, 0, 1:], labels[0, :-1]).astype(float) * beta[0]
            # bottom row
            # right
            ge[:, -1, :-1] += np.not_equal(comp[:, -1, :-1], labels[-1, 1:]).astype(float) * beta[0]
            # left
            ge[:, -1, 1:] += np.not_equal(comp[:, -1, 1:], labels[-1, :-1]).astype(float) * beta[0]

            # **********************************************************************************************************
            # direction 1 = 90° polar coord system
            ge[:, 1:-1, 1:-1] += (np.not_equal(comp[:, 1:-1, 1:-1], labels[1:-1, :-2]).astype(
                float)  # compare with above
                                  + np.not_equal(comp[:, 1:-1, 1:-1], labels[1:-1, 2:]).astype(float)) * beta[
                                     1]  # compare with below
            # left column
            # above
            ge[:, 1:, 0] += np.not_equal(comp[:, 1:, 0], labels[:-1, 0]).astype(float) * beta[1]
            # below
            ge[:, :-1, 0] += np.not_equal(comp[:, :-1, 0], labels[1:, 0]).astype(float) * beta[1]
            # right column
            # above
            ge[:, 1:, -1] += np.not_equal(comp[:, 1:, -1], labels[:-1, -1]).astype(float) * beta[1]
            # below
            ge[:, :-1, -1] += np.not_equal(comp[:, :-1, -1], labels[1:, -1]).astype(float) * beta[1]
            # top row
            # below
            ge[:, 0, :] += np.not_equal(comp[:, 0, :], labels[1, :]).astype(float) * beta[1]
            # bottom row
            # above
            ge[:, -1, :] += np.not_equal(comp[:, -1, :], labels[-2, :]).astype(float) * beta[1]

            # **********************************************************************************************************
            # direction 2 = 45° polar coord system
            if self.stencil is "8p":
                ge[:, 1:-1, 1:-1] += (np.not_equal(comp[:, 1:-1, 1:-1], labels[2:, :-2]).astype(
                    float)  # compare with right up
                                      + np.not_equal(comp[:, 1:-1, 1:-1], labels[:-2, 2:]).astype(float)) * beta[
                                         2]  # compare with left down
                # left column
                # right up
                ge[:, 1:, 0] += np.not_equal(comp[:, 1:, 0], labels[:-1, 1]).astype(float) * beta[2]
                # right column
                # left down
                ge[:, :-1, -1] += np.not_equal(comp[:, :-1, -1], labels[1:, -2]).astype(float) * beta[2]
                # top row
                # below left
                ge[:, 0, 1:] += np.not_equal(comp[:, 0, 1:], labels[1, :-1]).astype(float) * beta[2]
                # bottom row
                # above right
                ge[:, -1, :-1] += np.not_equal(comp[:, -1, :-1], labels[-2, 1:]).astype(float) * beta[2]
            # **********************************************************************************************************
            # direction 3 = 135° polar coord system
            if self.stencil is "8p":
                ge[:, 1:-1, 1:-1] += (np.not_equal(comp[:, 1:-1, 1:-1], labels[:-2, :-2]).astype(
                    float)  # compare with left up
                                      + np.not_equal(comp[:, 1:-1, 1:-1], labels[2:, 2:]).astype(float)) * beta[
                                         3]  # compare with right down
                # left column
                # right down
                ge[:, :-1, 0] += np.not_equal(comp[:, :-1, 0], labels[1:, 1]).astype(float) * beta[3]
                # right column
                # left up
                ge[:, 1:, -1] += np.not_equal(comp[:, 1:, -1], labels[:-1, -2]).astype(float) * beta[3]
                # top row
                # below right
                ge[:, 0, :-1] += np.not_equal(comp[:, 0, :-1], labels[1, 1:]).astype(float) * beta[3]
                # bottom row
                # above left
                ge[:, -1, 1:] += np.not_equal(comp[:, -1, 1:], labels[-2, :-1]).astype(float) * beta[3]

            # **********************************************************************************************************
            # overwrite corners
            # up left
            ge[:, 0, 0] = np.not_equal(comp[:, 0, 0], labels[1, 0]).astype(float) * beta[1] \
                          + np.not_equal(comp[:, 0, 0], labels[0, 1]).astype(float) * beta[0]
            if self.stencil is "8p":
                ge[:, 0, 0] += np.not_equal(comp[:, 0, 0], labels[1, 1]).astype(float) * beta[3]

            # low left
            ge[:, -1, 0] = np.not_equal(comp[:, -1, 0], labels[-1, 1]).astype(float) * beta[0] \
                           + np.not_equal(comp[:, -1, 0], labels[-2, 0]).astype(float) * beta[1]
            if self.stencil is "8p":
                ge[:, -1, 0] += np.not_equal(comp[:, -1, 0], labels[-2, 1]).astype(float) * beta[2]

            # up right
            ge[:, 0, -1] = np.not_equal(comp[:, 0, -1], labels[1, -1]).astype(float) * beta[1] \
                           + np.not_equal(comp[:, 0, -1], labels[0, -2]).astype(float) * beta[0]
            if self.stencil is "8p":
                ge[:, 0, -1] += np.not_equal(comp[:, 0, -1], labels[1, -2]).astype(float) * beta[2]

            # low right
            ge[:, -1, -1] = np.not_equal(comp[:, -1, -1], labels[-2, -1]).astype(float) * beta[1] \
                            + np.not_equal(comp[:, -1, -1], labels[-1, -2]).astype(float) * beta[0]
            if self.stencil is "8p":
                ge[:, -1, -1] += np.not_equal(comp[:, -1, -1], labels[-2, -2]).astype(float) * beta[3]

            # reshape and transpose gibbs energy, return
            return np.array([ge[l, :, :].ravel() for l in range(self.n_labels)]).T

        # ************************************************************************************************
        elif dim == 3:
            # TODO: [3D] implementation of gibbs energy
            raise Exception("3D not yet implemented.")

    def mcr(self, true_labels):
        """Compares classified with true labels for each iteration step (for synthetic data) to obtain a measure of
        mismatch/convergence."""
        mcr_vals = []
        n = len(true_labels)
        # TODO: [2D] implementation for MCR
        # TODO: [3D] implementation for MCR
        for label in self.labels:
            missclassified = np.count_nonzero(true_labels - label)
            mcr_vals.append(missclassified / n)
        return mcr_vals

    def get_std_from_cov(self, f, l):
        """
        Extracts standard deviation from covariance matrices for feature f and label l.
        :param f: feature (int)
        :param l: label (int)
        :return standard deviation from all covariance matrices for label/feature combination
        """
        stds = []
        for i in range(len(self.covs)):
            stds.append(np.sqrt(np.diag(self.covs[i][l])[f]))
        return stds

    def get_corr_coef_from_cov(self, l):
        """
        Extracts correlation coefficient from covariance matrix for label l.
        :param l: label (int)
        :retur: correlation coefficients from all covariance matrices for given label.
        """
        corr_coefs = []
        for i in range(len(self.covs)):
            corr_coef = self.covs[i][l, 0, 1]
            for f in [0, 1]:
                corr_coef = corr_coef / np.sqrt(np.diag(self.covs[i][l])[f])
            corr_coefs.append(corr_coef)
        return corr_coefs

    def plot_mu_stdev(self):
        """Plot mean and standard deviation over all iterations."""
        fig, ax = plt.subplots(nrows=self.n_feat, ncols=2, figsize=(15, 5 * self.n_feat))

        ax[0, 0].set_title(r"$\mu$")
        ax[0, 1].set_title(r"$\sigma$")

        for f in range(self.n_feat):
            for l in range(self.n_labels):
                if np.mean(np.array(self.mus)[:, :, f][:, l]) == -9999:
                    continue
                else:
                    ax[f, 0].plot(np.array(self.mus)[:, :, f][:, l], label="Label " + str(l))

            ax[f, 0].set_ylabel("Feature " + str(f))

            for l in range(self.n_labels):
                ax[f, 1].plot(self.get_std_from_cov(f, l), label="Label " + str(l))

        ax[f, 0].set_xlabel("Iterations")
        ax[f, 1].set_xlabel("Iterations")
        ax[f, 1].legend(loc=9, bbox_to_anchor=(0.5, -0.25), ncol=3)

        plt.show()

    def plot_acc_ratios(self, linewidth=1):
        """Plot acceptance ratios for beta, mu and covariance."""
        fig, ax = plt.subplots(ncols=3, figsize=(15, 4))

        ax[0].set_title(r"$\beta$")
        ax[0].plot(self.beta_acc_ratio, linewidth=linewidth, color="black")

        ax[1].set_title(r"$\mu$")
        ax[1].plot(self.mu_acc_ratio, linewidth=linewidth, color="red")

        ax[2].set_title("Covariance")
        ax[2].plot(self.cov_acc_ratio, linewidth=linewidth, color="indigo")

    def diagnostics_plot(self, true_labels=None, ie_range=None, transpose=False):
        """Diagnostic plots for analyzing convergence and segmentation results.


        Args:
            true_labels (:obj:`np.ndarray`):
            ie_range (:obj:`tuple` or :obj:`list`): Start and end point of iteration slice to used in the calculation
                of the information entropy.

        Returns:
            Plot
        """
        if true_labels is not None:
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(4, 2)
        else:
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(2, 2)

        rcParams.update({'font.size': 8})

        # plot beta
        ax1 = plt.subplot(gs[0, :-1])
        ax1.set_title(r"$\beta$")

        betas = np.array(self.betas)
        if self.dim == 1:
            ax1.plot(betas, label="beta", linewidth=1)
        else:
            for b in range(betas.shape[1]):
                ax1.plot(betas[:, b], label="beta " + str(b), linewidth=1)

        ax1.set_xlabel("Iterations")
        ax1.legend()

        # plot correlation coefficient
        ax2 = plt.subplot(gs[0, -1])
        ax2.set_title("Correlation coefficient")
        for l in range(self.n_labels):
            ax2.plot(self.get_corr_coef_from_cov(l), label="Label " + str(l), linewidth=1)
        ax2.legend()
        ax2.set_xlabel("Iterations")

        # 1D
        if self.dim == 1:
            # PLOT LABELS
            ax3 = plt.subplot(gs[1, :])
            ax3.imshow(np.array(self.labels), cmap=cmap, norm=cmap_norm, aspect='auto', interpolation='nearest')
            ax3.set_ylabel("Iterations")
            ax3.set_title("Labels")
            ax3.grid(False)  # disable grid

            if true_labels is not None:
                # plot the latent field
                ax4 = plt.subplot(gs[2, :])
                ax4.imshow(np.tile(np.expand_dims(true_labels, axis=1), 50).T,
                           cmap=cmap, norm=cmap_norm, aspect='auto', interpolation='nearest')
                ax4.set_title("Latent field")
                ax4.grid(False)

                # plot the mcr
                ax5 = plt.subplot(gs[3, :])
                ax5.plot(self.mcr(true_labels), color="black", linewidth=1)
                ax5.set_ylabel("MCR")
                ax5.set_xlabel("Iterations")

        # 2D
        elif self.dim == 2:
            if ie_range is None:  # use all
                a = 0
                b = -1
            else:  # use given range
                a = ie_range[0]
                b = ie_range[1]

            max_lp = labels_map(self.labels, r=(a, b))
            # print(max_lp)

            # PLOT LABELS
            ax3 = plt.subplot(gs[1, 0])
            ax3.set_title("Labels (MAP)")
            if transpose:
                max_lp_plot = np.array(max_lp.reshape(self.shape[0], self.shape[1])).T
            else:
                max_lp_plot = np.array(max_lp.reshape(self.shape[0], self.shape[1]))
            ax3.imshow(max_lp_plot, cmap=cmap, norm=cmap_norm, interpolation='nearest')
            ax3.grid(False)

            # PLOT INFORMATION ENTROPY
            ie = compute_ie(compute_labels_prob(np.array(self.labels[a:b])))  # calculate ie
            ax4 = plt.subplot(gs[1, 1])
            ax4.set_title("Information Entropy")
            if transpose:
                ie_plot = ie.reshape(self.shape[0], self.shape[1]).T
            else:
                ie_plot = ie.reshape(self.shape[0], self.shape[1])
            iep = ax4.imshow(ie_plot, cmap="viridis", interpolation='nearest')
            ax4.grid(False)
            plt.colorbar(iep)

        plt.show()

    def nearest_points(self, layer):
        lst = self.geo_data.interfaces.loc[np.where(self.geo_data.interfaces.formation_number == layer)[0]]
        lst = lst.reset_index(drop=True)

        # store points pairwise with their x and y coordinate
        points = []
        for k in range(len(lst)):
            points.append((lst.loc[k, 'X'], lst.loc[k, 'Y']))

        # calculate distances and storage with correspondig position
        distances = []
        pos = []
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                distances += [euclideanDistance(points[i], points[j])]  # stores distances between two points
                pos += [[i, j]]  # stores corresponding points

        # gives points with smallest distance
        final_points = pos[np.where(distances == min(distances))[0][0]]

        X = (lst.X.loc[final_points[0]] + lst.X.loc[final_points[1]]) / 2
        Y = (lst.Y.loc[final_points[0]] + lst.Y.loc[final_points[1]]) / 2

        # append mean point between the two nearest points
        points.append((X, Y))

        # calculate distance between mean point and all other values
        distances = []
        pos = []
        for i in range(len(points) - 1):
            distances += [euclideanDistance(points[i], points[-1])]  # stores distances between two points
            pos += [i]  # stores correspondig points

        # points which is nearest to mean point
        k = hq.nsmallest(3, distances)[2]
        #m = hq.nsmallest(4, distances)[3]

        # 3 nearest points in lst
        final_points.append(np.where(distances == k)[0][0])
        #final_points.append(np.where(distances == m)[0][0])

        u = np.sum(lst.loc[final_points, ['X', 'Y', 'Z','formation_number']], axis=1).values
        p = np.sum(self.geo_data.interfaces.loc[:, ['X', 'Y', 'Z','formation_number']], axis=1)
        p = p.sort_index()

        final_points2 = []
        for i in range(len(u)):
            final_points2.append(np.where(p == u[i])[0][0])



        return final_points2

    def normalize_feature_vectors(self):
        return (self.feat - np.mean(self.feat, axis=0).T) / np.std(self.feat, axis=0)


def labels_map(labels, r=None):
    if r is None:
        r = (0, -1)

    lp = compute_labels_prob(np.array(labels[r[0]:r[1]]))
    return np.argmax(lp, axis=0)


def draw_labels_vect(labels_prob):
    """Vectorized draw of the label for each elements respective labels probability.

    Args:
        labels_prob (:obj:`np.ndarray`): (n_elements x n_labels) ndarray containing the element-specific labels
            probabilites for each element.

    Returns:
        :obj:`np.array` : Flat array containing the newly drawn labels for each element.

    """
    # draw a random number between 0 and 1 for each element
    r = np.random.rand(len(labels_prob))
    # cumsum labels probabilities for each element
    p = np.cumsum(labels_prob, axis=1)
    # calculate difference between random draw and cumsum probabilities
    d = (p.T - r).T
    # compare and count to get label
    return np.count_nonzero(np.greater_equal(0, d), axis=1)


def evaluate(log_target_prop, log_target_prev):
    ratio = np.exp(np.longfloat(log_target_prop - log_target_prev))

    if (ratio > 1) or (np.random.uniform() < ratio):
        return True, ratio  # if accepted

    else:
        return False, ratio  # if rejected


def _propose_cov(cov_prev, n_feat, n_labels, cov_jump_length, theta_jump_length):
    """Proposes a perturbed n-dimensional covariance matrix based on an existing one and a covariance jump length and
    theta jump length parameter.

    Args:
        cov_prev (:obj:`np.ndarray`): Covariance matrix.
        n_feat (int): Number of features.
        n_labels (int): Number of labels.
        cov_jump_length (float): Hyperparameter
        theta_jump_length (float): Hyperparameter

    Returns:
        :obj:`np.ndarray` : Perturbed covariance matrix.

    """
    # do svd on the previous covariance matrix
    comb = list(combinations(range(n_feat), 2))
    n_comb = len(comb)
    theta_jump = multivariate_normal(mean=[0 for i in range(n_comb)], cov=np.ones(n_comb) * theta_jump_length).rvs()

    if n_comb == 1:  # turn it into a list if there is only one combination (^= 2 features)
        theta_jump = [theta_jump]

    cov_prop = np.zeros_like(cov_prev)
    # print("cov_prev:", cov_prev)

    # loop over all labels (=layers of the covariance matrix)
    for l in range(n_labels):
        v_l, d_l, v_l_t = np.linalg.svd(cov_prev[l, :, :])
        # print("v_l:", v_l)
        # generate d jump
        log_d_jump = multivariate_normal(mean=[0 for i in range(n_feat)], cov=np.eye(n_feat) * cov_jump_length).rvs()
        # sum towards d proposal
        # if l == 0:
        d_prop = np.diag(np.exp(np.log(d_l) + log_d_jump))
        # else:
        #    d_prop = np.vstack((d_prop, np.exp(np.log(d_l) + np.log(d_jump))))
        # now tackle generating v jump
        a = np.eye(n_feat)
        # print("a init:", a)
        # print("shape a:", np.shape(a))
        for val in range(n_comb):
            rotation_matrix = _cov_proposal_rotation_matrix(v_l[:, comb[val][0]], v_l[:, comb[val][1]], theta_jump[val])
            # print("rot mat:", rotation_matrix)
            a = rotation_matrix @ a
            # print("a:", a)
        # print("v_l:", np.shape(v_l))
        v_prop = a @ v_l  # np.matmul(a, v_l)
        # print("d_prop:", d_prop)
        # print("v_prop:", np.shape(v_prop))
        cov_prop[l, :, :] = v_prop @ d_prop @ v_prop.T  # np.matmul(np.matmul(v_prop, d_prop), v_prop.T)
        # print("cov_prop:", cov_prop)

    return cov_prop


def _cov_proposal_rotation_matrix(x, y, theta):
    """Creates the rotation matrix needed for the covariance matrix proposal step.

    Args:
        x (:obj:`np.array`): First base vector.
        y (:obj:`np.array`): Second base vector.
        theta (float): Rotation angle.

    Returns:
        :obj:`np.ndarray` : Rotation matrix for covariance proposal step.

    """
    x = np.array([x]).T
    y = np.array([y]).T

    uu = x / np.linalg.norm(x)
    vv = y - uu.T @ y * uu
    vv = vv / np.linalg.norm(vv)
    # what is happening

    # rotation_matrix = np.eye(len(x)) - np.matmul(uu, uu.T) - np.matmul(np.matmul(vv, vv.T) + np.matmul(np.hstack((uu, vv)), np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])), np.hstack((uu, vv)).T)
    rotation_matrix = np.eye(len(x)) - uu @ uu.T - vv @ vv.T + np.hstack((uu, vv)) @ np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.hstack((uu, vv)).T
    return rotation_matrix


def _calc_labels_prob(te, t):
    """"Calculate labels probability for array of total energies (te) and totally arbitrary skalar value t."""
    return (np.exp(-te / t).T / np.sum(np.exp(-te / t), axis=1)).T


def pseudocolor(shape, stencil=None):
    """Graph coloring based on the physical dimensions for independent labels draw.

    Args:
        extent (:obj:`tuple` of int): Data extent in (Y), (Y,X) or (Y,X,Z) for 1D, 2D or 3D respectively.
        stencil:

    Returns:

    """
    dim = len(shape) - 1
    # ************************************************************************************************
    # 1-DIMENSIONAL
    if dim == 1:
        i_w = np.arange(0, shape[0], step=2)
        i_b = np.arange(1, shape[0], step=2)

        return np.array([i_w, i_b]).T

    # ************************************************************************************************
    # 2-DIMENSIONAL
    elif dim == 2:
        if stencil is None or stencil == "8p":
            # use 8 stamp as default, resulting in 4 colors
            colors = 4
            # color image
            colored_image = np.tile(np.kron([[0, 1], [2, 3]] * int(shape[0] / 2), np.ones((1, 1))), int(shape[1] / 2))
            colored_flat = colored_image.reshape(shape[0] * shape[1])

            # initialize storage array
            ci = []
            for c in range(colors):
                x = np.where(colored_flat == c)[0]
                ci.append(x)
            return np.array(ci)

        elif stencil == "4p":
            # use 4 stamp, resulting in 2 colors (checkerboard)
            colors = 2
            # color image
            colored_image = np.tile(np.kron([[0, 1], [1, 0]] * int(shape[0] / 2), np.ones((1, 1))), int(shape[1] / 2))
            colored_flat = colored_image.reshape(shape[0] * shape[1])

            # initialize storage array
            ci = []
            for c in range(colors):
                x = np.where(colored_flat == c)[0]
                ci.append(x)
            return ci
        else:
            raise Exception(" In 2D space the stamp parameter needs to be either None (defaults to 8p), 4p or 8p.")

    # ************************************************************************************************
    # 3-DIMENSIONAL
    elif dim == 3:
        raise Exception("3D space not yet supported.")
        # TODO: 3d graph coloring


def bic(feat_vector, n_labels, plot):
    """Plots the Bayesian Information Criterion of Gaussian Mixture Models for the given features and range of labels
    defined by the given upper boundary.

    Args:
        feat_vector (:obj:`np.ndarray`): Feature vector containing the data in a flattened format.
        n_labels (int): Sets the included upper bound for the number of features to be considered in the analysis.

    Returns:
        Plot

    """
    n_comp = np.arange(1, n_labels + 1)
    # create array of GMMs in range of components/labels and fit to observations
    gmms = np.array([mixture.GaussianMixture(n_components=n, covariance_type="full").fit(feat_vector) for n in n_comp])
    # calculate BIC for each GMM based on observartions
    bics = np.array([gmm.bic(feat_vector) for gmm in gmms])
    # take sequential difference
    # bic_diffs = np.ediff1d(bics)

    # find index of minimum BIC
    # bic_min = np.argmin(bics)
    # bic_diffs_min = np.argmin(np.abs(bic_diffs))

    # d = np.abs(bic_diffs[bic_diffs_min] * d_factor)
    bic_min = np.argmin(bics)

    # do a nice plot so the user knows intuitively whats happening
    if plot == True:
        fig = plt.figure()  # figsize=(10, 10)
        plt.plot(n_comp, bics, label="bic")
        plt.plot(n_comp[bic_min], bics[bic_min], "ko")
        plt.title("Bayesian Information Criterion")
        plt.xlabel("Number of Labels")
        plt.axvline(n_comp[bic_min], color="black", linestyle="dashed", linewidth=0.75)
        plt.show()

        print("global minimum: ", n_comp[bic_min])
    return n_comp[bic_min]


def gibbs_comp_f(a, value):
    """Helper function for the Gibbs energy calculation using Scipy's generic filter function."""
    a = a[a != -999.]
    return np.count_nonzero(a != value)


def prepare_gempy_input(input_data, result_data):
    """Prepares the data for the input in Gempy and puts out a csv-file including the data"""
    # data convertion to gempy

    gempy = pd.DataFrame({'X': input_data.X, 'Y': input_data.Y, 'Z': input_data.Z, 'formation': result_data,
                          'borehole': input_data['Well Name']})

    for k in range(0, len(gempy) - 1):
        if gempy.loc[k, 'formation'] == gempy.loc[k + 1, 'formation']:
            gempy = gempy.drop(k)

    gempy.index = range(len(gempy))

    for k in range(0, 1 + len(set(list(gempy['formation'])))):
        gempy['formation'] = gempy['formation'].replace(to_replace=k, value='Layer%d' % (k))

    for k in range(0, 1 + len(set(list(gempy['formation'])))):
        gempy['formation'] = gempy['formation'].replace(to_replace=k, value='Layer%d' % (k))

    gempy.to_csv('../data/Gempy_Simple_4_layer_90degrees.csv', index=False)


def normalize_feature_vectors(self):
    return (self - np.mean(self, axis=0).T) / np.std(self, axis=0)


def test_bic(feature_vectors_norm,n):
    # Bayesian information criteria
    nft = []
    for k in range(2, n):
        nft.append(bic(feature_vectors_norm, k,
                       plot=False))  # Investigate the number of labels (one label can include several cluster)
    nf = max(set(nft), key=nft.count)  # put out the most common value in bic
    print('The optimal number of layers is: ', nf)
    return nf, nft

def euclideanDistance(coordinate1, coordinate2):
    return pow(pow(coordinate1[0] - coordinate2[0], 2) + pow(coordinate1[1] - coordinate2[1], 2), .5)