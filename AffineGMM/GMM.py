from functools import reduce
from typing import Union
from typing import Iterable
import numpy as np
from scipy.stats import multivariate_normal as mvn
import scipy.sparse as ss
from .Gaussian import *
from .UniGMM import UniGMM


class GMM:
    def __init__(self, means: np.ndarray, covariances: Union[np.ndarray, ss.csr_matrix], weights: np.ndarray):
        """
        Initialize GMM object.

        :param means: c × dim  mean vector
        :param covariances: c × dim × dim covariance matrix
        :param weights: c × 1 weights
        """
        self.mu = means
        self.sigma = covariances
        self.weights = weights.reshape(-1, 1)

    def __str__(self):
        return f"GMM with {self.mu.shape[0]} components and {self.mu.shape[1]} dimensions"

    def __repr__(self):
        return f"GMM with {self.mu.shape[0]} components and {self.mu.shape[1]} dimensions"

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            if other != 0.:
                return GMM(self.mu * other, other ** 2 * self.sigma, self.weights)
            else:
                return GMM(np.zeros((1, self.mu.shape[1])), np.eye(self.mu.shape[1]), np.ones(1, self.mu.shape[1]))
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        raise NotImplementedError

    def __rmatmul__(self, other):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, GMM):
            raise NotImplementedError
        elif isinstance(other, float) or isinstance(other, int):
            return GMM(self.mu + other, self.sigma, self.weights)
        elif isinstance(other, np.ndarray):
            return GMM(self.mu + other.reshape(1, self.mu.shape[1]), self.sigma, self.weights)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, GMM):
            new_weights = self.weights.reshape(-1, 1) @ other.weights.reshape(1, -1)
            new_mu = self.mu.reshape(-1, 1) - other.mu.reshape(1, -1)
            new_sigma = np.sqrt((self.sigma ** 2).reshape(-1, 1) + (other.sigma ** 2).reshape(1, -1))
            return GMM(new_mu.flatten(), new_sigma.flatten(), new_weights.flatten())
        elif isinstance(other, float) or isinstance(other, int):
            return GMM(self.mu - other, self.sigma, self.weights)
        elif isinstance(other, np.ndarray):
            temp = other.reshape(-1, 1)
            if self.mu.shape[1] != other.shape[0]:
                raise ValueError
            else:
                temp_mu = self.mu.copy()
                for i in range(self.mu.shape[0]):
                    temp_mu[i, :] -= other
                return GMM(temp_mu, self.sigma, self.weights)
        else:
            raise NotImplementedError

    def __neg__(self):
        return GMM(-self.mu, self.sigma, self.weights)

    def __rsub__(self, other):
        if isinstance(other, GMM):
            new_weights = self.weights.reshape(-1, 1) @ other.weights.reshape(1, -1)
            new_mu = -self.mu.reshape(-1, 1) + other.mu.reshape(1, -1)
            new_sigma = np.sqrt((self.sigma ** 2).reshape(-1, 1) + (other.sigma ** 2).reshape(1, -1))
            return GMM(new_mu.flatten(), new_sigma.flatten(), new_weights.flatten())
        elif isinstance(other, float) or isinstance(other, int):
            return GMM(-self.mu, self.sigma, self.weights) + other
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self * (1 / other)
        else:
            raise NotImplementedError

    def aic(self, x):
        raise NotImplementedError

    def bic(self, x):
        raise NotImplementedError

    @staticmethod
    def from_scikit_learn(gm):
        weights = gm.weights_.reshape(-1, 1)
        means = gm.means_
        covariances = gm.covariances_
        cov_type = gm.covariance_type
        if cov_type == 'tied':
            covariances = np.array([covariances for _ in range(means.shape[0])])
        elif cov_type == 'diag':
            covariances = np.array([np.diag(covariances[i, :]) for i in range(covariances.shape[0])])
        elif cov_type == 'spherical':
            raise NotImplementedError
        elif cov_type == 'full':
            pass
        else:
            raise ValueError(f"Covariance type {cov_type} mismatch")
        return GMM(means, covariances, weights)

    def mean(self):
        """
        Return mean value of GMM

        :return: 1 × dim mean value
        """
        return self.weights.T @ self.mu

    def covariance(self):
        """
        Return covariance matrix of GMM

        :return: dim × dim covariance matrix
        """
        m = self.mean()
        s = -np.outer(m, m)
        for i in range(self.mu.shape[0]):
            cm = self.mu[i, :]
            cvar = self.sigma[i, :, :]
            s += self.weights[i, 0] * (np.outer(cm, cm) + cvar)
        return s

    def score(self, x):
        raise NotImplementedError

    def linear_transform(self, other, copy=True):
        """
        Calculate linear transformation of a GMM x (y = Hx)

        :param other: transformation matrix H
        :param copy: if return a new GMM
        :return: y = Hx
        """
        if isinstance(other, np.ndarray):
            tmp0 = np.einsum('ij,kj->ki', other, self.mu)
            tmp1 = np.einsum('ij,rjl->ril', other, self.sigma)
            tmp2 = np.einsum('rij,lj->ril', tmp1, other)
            if copy:
                return GMM(tmp0, tmp2, self.weights)
            else:
                self.mu = tmp0
                self.sigma = tmp2
        elif isinstance(other, ss.csr_array) or isinstance(other, ss.csc_array) or isinstance(other,
                                                                                              ss.csr_matrix) or isinstance(
            other, ss.csc_matrix):
            # TODO: Try to implement it using sparse style
            tmp0 = np.einsum('ij,kj->ki', other, self.mu)
            tmp1 = np.einsum('ij,rjl->ril', other, self.sigma)
            tmp2 = np.einsum('rij,lj->ril', tmp1, other)
            if copy:
                return GMM(tmp0, tmp2, self.weights)
        else:
            raise NotImplementedError('Use np.ndarray or scipy.sparse.csr_matrix')

    def inverse_linear_transformation(self, other, copy=True):
        """
        Calculate linear transformation of a GMM x (y = H^{-1}x)

        :param other: transformation matrix H
        :param copy: if return a new GMM
        :return: y = H^{-1} x
        """
        if isinstance(other, np.ndarray):
            num_of_features = self.mu.shape[0]
            tmp_mu = np.linalg.solve(other, self.mu.T).T
            tmp_sigma = np.zeros_like(self.sigma)
            for i in range(num_of_features):
                tmp_sigma[i, :, :] = np.linalg.solve(other, self.sigma[i, :, :]) @ np.linalg.solve(other.T, np.eye(
                    self.sigma[i, :, :].shape[0]))
            if copy:
                return GMM(tmp_mu, tmp_sigma, self.weights)
            else:
                self.mu = tmp_mu
                self.sigma = tmp_sigma
        else:
            raise NotImplementedError

    def affine(self, a, b, copy=True):
        """
        Affine transform of GMM (y = aX + b)

        :param a: Linear transform matrix
        :param b: affine vector
        :param copy: generate a new GMM or not
        :return: aX + b Affine transformation
        """
        if copy:
            return self.linear_transform(a, copy=True) + b
        else:
            self.linear_transform(a, copy=False)
            self.mu += b

    def pdf(self, x: Union[np.ndarray, Iterable, int, float]):
        """
        Calculate probability density function (PDF) of GMM

        :param x: n × dim Sample points
        :return: 1 × dim PDFs
        """
        if self.mu.shape[1] > 1:
            return self.weights.T @ multiple_pdf_vec_input(x, self.mu, self.sigma)
        else:
            return self.weights.T @ single_pdf_vec_input(x, self.mu, self.sigma)

    def cdf(self, x: Union[np.ndarray, Iterable, int, float], approx=True):
        """
        Calculate cumulative density function (CDF) of GMM
        :param x: n × dim Sample points
        :param approx: bool Using fast approximation of CDF of multivariate Gaussian distributions
        :return: 1 × dim CDFs
        """
        return self.weights.T @ multiple_cdf_vec_input(x, self.mu, self.sigma, approx=approx)

    def sample(self, num_of_samples=1):
        """
        Sample from GMM

        :param num_of_samples: number of samples
        :return: data: samples
        """
        if num_of_samples < 1:
            raise ValueError(
                f"Invalid value for 'n_samples': {num_of_samples} . The sampling requires at "
                "least one sample."
            )
        components = np.random.choice(self.mu.shape[0], size=int(num_of_samples), p=self.weights.flatten())
        components_index = [np.nonzero(components == i)[0] for i in range(len(self.mu))]
        del components
        data = np.zeros((num_of_samples, self.mu.shape[1]))
        for i, c in enumerate(components_index):
            data[c] = np.random.multivariate_normal(mean=self.mu[i, :], cov=self.sigma[i, :, :], size=len(c))
        del components_index
        return data

    def normalization(self):
        if np.sum(self.weights) != 1.0:
            self.weights /= np.sum(self.weights)

    def reduce_components(self, tol=1e-6, copy=True):
        """
        Reduce GMM components with a given tolerance

        :param tol:
        :param copy:
        :return:
        """
        saved_index = np.nonzero(self.weights >= tol)[0]
        if copy:
            new_weight = self.weights[saved_index, :] / np.sum(self.weights[saved_index])
            return GMM(self.mu[saved_index, :], self.sigma[saved_index, :, :], new_weight)
        else:
            self.mu = self.mu[saved_index, :]
            self.sigma = self.sigma[saved_index, :, :]
            self.weights = self.weights[saved_index] / np.sum(self.weights[saved_index])

    def abs_risk(self, threshold=10.):
        return 1 - self.cdf(np.abs(threshold)) + self.cdf(-np.abs(threshold))

    def extract_feature(self, index_of_feature: int):
        return UniGMM(mu=self.mu[:, index_of_feature], sigma=np.sqrt(self.sigma[:, index_of_feature, index_of_feature]),
                      weights=self.weights)

    def extract_two_features(self, feature1, feature2):
        """
        Extracts a 2D GMM from the original GMM object based on the specified features.

        :param feature1: index of the first feature to use
        :param feature2: index of the second feature to use
        :return: a new GMM object with means, covariances, and weights based on the specified features
        """
        return self.extract_features([feature1, feature2])

    def extract_features(self, feature_indices):
        """
        Extract features corresponding to the given indices and return a new GMM object.

        :param feature_indices: a list or 1-D array of feature indices to extract
        :return: a new GMM object with the extracted features
        """
        mu = self.mu[:, feature_indices]
        sigma = self.sigma[:, feature_indices][:, :, feature_indices]
        return GMM(mu, sigma, self.weights)

    def inverse_cdf(self, p):
        """
        Compute the inverse cumulative distribution function (quantile function) of the GMM for a given probability.

        :param p: target probability
        :return: dim × 1 input point corresponding to the quantile
        """
        raise NotImplementedError
