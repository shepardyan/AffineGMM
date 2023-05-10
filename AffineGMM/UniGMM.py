from typing import Union
from typing import Iterable
import numpy as np
from scipy.special import erf
from .Gaussian import *
import cmath  # Import the complex math modul
import sympy as sp


def solve_quartic(coeffs):
    roots = np.roots(coeffs)
    # Filter out the imaginary parts of the roots
    real_roots = [root.real for root in roots if root.imag == 0]
    # Return the real roots as a list
    return real_roots


def gaussian_pdf(x, mu, sigma):
    if sigma == 0.0:
        res = np.zeros_like(x)
        res[x == mu] = np.inf
        return res
    else:
        return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def gaussian_cdf(x, mu, sigma):
    if sigma == 0.0:
        return np.sign(x - mu)
    else:
        return 0.5 * (1.0 + erf((x - mu) / (sigma * np.sqrt(2.0))))


def gaussian_approx_cdf(xs, mean, var):
    h0, h1, h2, h3, h4 = -0.0005, 0.424, -0.06211, -0.02918, 0.00709

    xm = (xs - mean) / var

    def _pwl_cdf(x):
        if x > 3:
            return 1
        elif x < -3:
            return 0
        elif x < 0:
            return 0.5 - (h4 * x ** 4 - h3 * x ** 3 + h2 * x ** 2 - h1 * x + h0)
        else:
            return 0.5 + (h4 * x ** 4 + h3 * x ** 3 + h2 * x ** 2 + h1 * x + h0)

    return np.vectorize(_pwl_cdf)(xm)


class UniGMM:
    def __init__(self, mu, sigma, weights):
        self.mu = np.array(mu).reshape(-1)
        self.sigma = np.array(sigma).reshape(-1)
        self.weights = np.array(weights).reshape(-1)
        assert self.mu.shape == self.sigma.shape
        assert self.mu.shape == self.weights.shape

    def __str__(self):
        return f"Univariate GMM with {len(self.mu)} components"

    def __repr__(self):
        return f"Univariate GMM with {len(self.mu)} components"

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            if other != 0.:
                return UniGMM(np.array(self.mu) * other, np.sqrt(np.array(self.sigma) ** 2 * (other ** 2)),
                              weights=self.weights)
            else:
                return UniGMM([0.], [0.], [1.])
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, UniGMM):
            new_weights = self.weights.reshape(-1, 1) @ other.weights.reshape(1, -1)
            new_mu = self.mu.reshape(-1, 1) + other.mu.reshape(1, -1)
            new_sigma = np.sqrt((self.sigma ** 2).reshape(-1, 1) + (other.sigma ** 2).reshape(1, -1))
            return UniGMM(new_mu.flatten(), new_sigma.flatten(), new_weights.flatten())
        elif isinstance(other, float) or isinstance(other, int):
            return UniGMM(self.mu + other, self.sigma, self.weights)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, UniGMM):
            new_weights = self.weights.reshape(-1, 1) @ other.weights.reshape(1, -1)
            new_mu = self.mu.reshape(-1, 1) - other.mu.reshape(1, -1)
            new_sigma = np.sqrt((self.sigma ** 2).reshape(-1, 1) + (other.sigma ** 2).reshape(1, -1))
            return UniGMM(new_mu.flatten(), new_sigma.flatten(), new_weights.flatten())
        elif isinstance(other, float) or isinstance(other, int):
            return UniGMM(self.mu - other, self.sigma, self.weights)
        else:
            raise NotImplementedError

    def __neg__(self):
        return UniGMM(-self.mu, self.sigma, self.weights)

    def __rsub__(self, other):
        if isinstance(other, UniGMM):
            new_weights = self.weights.reshape(-1, 1) @ other.weights.reshape(1, -1)
            new_mu = -self.mu.reshape(-1, 1) + other.mu.reshape(1, -1)
            new_sigma = np.sqrt((self.sigma ** 2).reshape(-1, 1) + (other.sigma ** 2).reshape(1, -1))
            return UniGMM(new_mu.flatten(), new_sigma.flatten(), new_weights.flatten())
        elif isinstance(other, float) or isinstance(other, int):
            return -UniGMM(self.mu, self.sigma, self.weights) + other
        else:
            raise NotImplementedError

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return self * (1 / other)
        else:
            raise NotImplementedError

    def pdf(self, x: Union[np.ndarray, Iterable, int, float]):
        """
        计算高斯混合模型的概率密度函数
        :param x: 需要计算的点
        :return:
        """
        return self.weights @ multiple_pdf_vec_input(x.reshape(-1, 1), self.mu.reshape(-1, 1),
                                                     (self.sigma ** 2).reshape(-1, 1, 1))

    def cdf(self, x: Union[np.ndarray, Iterable, int, float]):
        cdf_data = self.weights[0] * gaussian_cdf(x, self.mu[0], self.sigma[0])
        if len(self.mu) > 1:
            for i in range(1, len(self.mu)):
                cdf_data += self.weights[i] * gaussian_cdf(x, self.mu[i], self.sigma[i])
        return cdf_data

    def sample(self, num_of_samples=1):
        """
        从高斯混合模型中采样
        :param num_of_samples: 采样数
        :return: data: 样本
        """
        if num_of_samples < 1:
            raise ValueError(
                "Invalid value for 'n_samples': %d . The sampling requires at "
                "least one sample."
            )

        if self.sigma[0] == 0:
            return np.ones(int(num_of_samples))
        else:
            components = np.random.choice(len(self.mu), size=int(num_of_samples), p=self.weights)
            components_index = [np.nonzero(components == i)[0] for i in range(len(self.mu))]
            del components
            data = np.zeros(num_of_samples)
            for i, c in enumerate(components_index):
                data[c] = np.random.normal(loc=self.mu[i], scale=self.sigma[i], size=len(c))
            del components_index
            return data

    def normalization(self):
        if np.sum(self.weights) != 1.0:
            self.weights /= np.sum(self.weights)

    def reduce_components(self, tol=1e-6, copy=True):
        """
        根据给定阈值减少UniGMM的分量
        :param tol:
        :param copy:
        :return:
        """
        saved_index = np.nonzero(self.weights >= tol)[0]
        if copy:
            new_weight = self.weights[saved_index] / np.sum(self.weights[saved_index])
            return UniGMM(self.mu[saved_index], self.sigma[saved_index], new_weight)
        else:
            self.mu = self.mu[saved_index]
            self.sigma = self.sigma[saved_index]
            self.weights = self.weights[saved_index] / np.sum(self.weights[saved_index])

    def abs_risk(self, threshold=10.):
        return 1 - self.cdf(np.abs(threshold)) + self.cdf(-np.abs(threshold))

    def approx_cdf(self, x: Union[np.ndarray, Iterable, int, float]):
        cdf_data = self.weights[0] * gaussian_approx_cdf(x, self.mu[0], self.sigma[0])
        if len(self.mu) > 1:
            for i in range(1, len(self.mu)):
                cdf_data += self.weights[i] * gaussian_approx_cdf(x, self.mu[i], self.sigma[i])
        return cdf_data

    def interval(self, threshold=1):
        h0, h1, h2, h3, h4 = -0.0005, 0.424, -0.06211, -0.02918, 0.00709

        def _pwl_cdf_ub(x):
            return 0.5 + (h4 * x ** 4 + h3 * x ** 3 + h2 * x ** 2 + h1 * x + h0)

        def _pwl_cdf_lb(x):
            return 0.5 - (h4 * x ** 4 - h3 * x ** 3 + h2 * x ** 2 - h1 * x + h0)

        xx = sp.Symbol('x')
        xm = (xx - self.mu[0]) / self.sigma[0]
        result_ub = self.weights[0] * _pwl_cdf_ub(xm)
        result_lb = self.weights[0] * _pwl_cdf_lb(xm)
        if self.weights.shape[0] > 1:
            for i in range(1, self.weights.shape[0]):
                xm = (xx - self.mu[i]) / self.sigma[i]
                result_ub += self.weights[i] * _pwl_cdf_ub(xm)
                result_lb += self.weights[i] * _pwl_cdf_lb(xm)
        coefficients_ub = sp.Poly(result_ub, xx).coeffs()  # type:list
        coefficients_lb = sp.Poly(result_lb, xx).coeffs()  # type:list
        coefficients_lb[-1] -= 1 - threshold
        coefficients_ub[-1] -= threshold

        real_roots_lb = solve_quartic(coefficients_lb)
        real_roots_ub = solve_quartic(coefficients_ub)
        return min(real_roots_lb), max(real_roots_ub)
