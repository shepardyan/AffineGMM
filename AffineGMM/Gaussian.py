from scipy.stats import multivariate_normal as mvnormal
from scipy.special import erf, erfc
from scipy.stats import mvn
import numpy as np
from approxcdf import mvn_cdf, bvn_cdf

_LOG_2_PI = np.log(2 * np.pi)


def multiple_pdf_vec_input(xs, means, covs):
    """
    Calculate PDFs with multiple parameters

    :param xs: n × dim Vectorized sample input
    :param means: c × dim  mean vector
    :param covs:  c × dim × dim covariance matrix
    :return: PDFs: c × dim PDFs
    """
    vals, vecs = np.linalg.eigh(covs)
    logdets = np.sum(np.log(vals), axis=1)
    valsinvs = np.divide(1., vals)
    Us = vecs * np.sqrt(valsinvs)[:, None]
    devs = xs[:, np.newaxis, :] - means[np.newaxis, :, :]
    mahas = np.sum(np.square(np.einsum('jnk,nki->jni', devs, Us)), axis=2)
    return np.exp(-0.5 * (xs.shape[1] * _LOG_2_PI + mahas + logdets[np.newaxis, :])).T


def multiple_cdf_vec_input(xs, means, covs, approx=True):
    """
    Calculate CDFs with multiple parameters

    :param xs: n × dim Vectorized sample input
    :param means: c × dim  mean vector
    :param covs:  c × dim × dim covariance matrix
    :return: CDFs: c × dim CDFs
    """
    return np.array([single_cdf_vec_input(xs, means, covs, i, approx=approx) for i in range(means.shape[0])])


def single_cdf_vec_input(xs, means, covs, index, approx=True):
    """
    Calculate single CDF with indexed parameter

    :param xs: n × dim Vectorized sample input
    :param means: c × dim  mean vector
    :param covs:  c × dim × dim covariance matrix
    :param index: integer index
    :param approx: bool Using fast approximation or not
    :return: PDFs: c × dim PDFs
    """
    if approx:
        if means.shape[1] == 2:
            return bvn_cdf(xs, covs[index, :, :], means[index, :])
        else:
            return mvn_cdf(xs, covs[index, :, :], means[index, :])
    else:
        return mvnormal.cdf(xs, means[index, :], covs[index, :, :], allow_singular=True)


def single_pdf_vec_input(xs, means, covs):
    """
    Calculate single CDF with indexed parameter

    :param xs: n × dim Vectorized sample input
    :param means: c × dim  mean vector
    :param covs:  c × dim × dim covariance matrix
    :param index: integer index
    :return: PDFs: c × dim PDFs
    """
    return np.array([mvnormal.pdf(xs, means[i, :], covs[i, :, :], allow_singular=True) for i in range(means.shape[0])])


def univariate_cdf_vec_input(xs, mean, var):
    sigma = np.sqrt(var)
    if sigma == 0.0:
        return np.sign(xs - mean)
    else:
        return 0.5 * (1.0 + erf((xs - mean) / (sigma * np.sqrt(2.0))))


def multivariate_gaussian_pdf(x, mean, cov):
    """

    :param x: n × dim sample matrix
    :param mean: 1 × dim mean vector
    :param cov: dim × dim covariance matrix
    :return: PDF: 1 × dim PDF
    """
    return multiple_pdf_vec_input(x, mean.reshape(1, -1), cov.reshape(1, cov.shape[0], cov.shape[1]))
