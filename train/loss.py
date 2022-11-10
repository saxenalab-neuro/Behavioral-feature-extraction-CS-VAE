##############import#################
import numpy as np
import tensorflow as tf
tf.random.set_seed(33)
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Layer
import math

##################define classes for loss and regularizor######
def Distance(x, y):
    d = tf.reduce_sum(tf.square(x - y), 1)
    dx = tf.maximum(d, 1e-9)
    dist = tf.sqrt(dx)
    return tf.reduce_mean(dist)



def swiss_roll(size, noise=1.5):
    u = K.random_uniform(shape=(size[0], 1), dtype="float32")
    u2 = K.random_uniform(shape=(size[0], 1), dtype="float32") * 10

    t = 1.5 * np.pi * (1 + 1.5 * u)
    x = t * K.cos(t)  # +11*K.ones_like(u2)
    y = u2  # + 11*K.ones_like(u2)
    z = t * K.sin(t)  # + 11*K.ones_like(u2)

    X = K.concatenate([x, z], axis=-1)

    return X


def squre_roll(size, noise=1.5):
    u = K.random_uniform(shape=(size[0], 1), dtype="float32")
    u2 = K.random_uniform(shape=(size[0], 1), dtype="float32") * 10
    #     print(u)
    t = 1.5 * np.pi * (1 + 100 * u)
    x = t * K.cos(t)  # +11*K.ones_like(u2)
    y = u2  # + 11*K.ones_like(u2)
    z = t * K.sin(t)  # + 11*K.ones_like(u2)

    X = K.concatenate([x, y, z], axis=-1)

    return X


def spherical(size, ndim=3):
    vec = K.random_normal(shape=(size[0], ndim), dtype="float32")
    #     vec /= K.linalg.norm(vec, axis=0)
    return vec


def clusterdis(size, noise=1.5):
    a = int(size[0] / 4)
    b = a
    c = a
    d = size[0] - a - b - c
    u = K.random_uniform(shape=(a, 3), dtype="float32")
    u2 = K.random_uniform(shape=(b, 3), dtype="float32") * 2
    u3 = K.random_uniform(shape=(c, 3), dtype="float32") * 4
    u4 = K.random_uniform(shape=(d, 3), dtype="float32") * 8
    #     sess = tf.Session()
    #     with sess.as_default():
    #         t=K.eval(size[0])
    X = K.concatenate([u, u2, u3, u4], axis=0)

    #     X,_=make_blobs(n_samples=t, centers=4, n_features=3,random_state=42)
    return X


def GMM(size, noise=1.5):
    N, D = size[0], 3  # number of points and dimenstinality

    means = np.array([[0.5, 0.0, 0.0],
                      [0.0, 0.5, 0.8],
                      [-0.5, -0.5, -0.5],
                      [-0.8, 0.3, 0.4]])
    #     means= K.constant(means)

    covs = np.array([np.diag([0.01, 0.01, 0.01]),
                     np.diag([0.01, 0.01, 0.01]),
                     np.diag([0.01, 0.01, 0.01]),
                     np.diag([0.01, 0.01, 0.01])])
    #     covs=K.constant(covs)

    n_gaussians = means.shape[0]

    points = []
    if N % 4 == 0:
        L = int(N / 4)

    for i in range(len(means)):
        x = np.random.multivariate_normal(means[i], covs[i], L)
        points.append(x)

    points = K.constant(np.concatenate(points))

    return points


def get_kernel(X, Z, ksize):
    #     print(X.shape,Z.shape)
    G = K.sum((K.expand_dims(X, axis=1) - Z) ** 2, axis=-1)  # Gram matrix
    G = K.exp(-G / (ksize)) / (math.sqrt(2 * np.pi * ksize) * K.ones_like(-G / (ksize)))
    return G


class DiagonalWeight(Constraint):
    """Constrains the weights to be diagonal.
    """

    def __init__(self, N):
        self.m = K.eye(N)

    def __call__(self, w):
        #         N = K.int_shape(w)[-1]
        #         m = K.eye(N)
        w = w * self.m
        return w


class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = -0.5 * K.sum(1 + log_var -
                                K.square(mu) - K.exp(log_var), axis=-1)

        self.add_loss(tf.reduce_mean(kl_batch), inputs=inputs)

        return inputs, tf.reduce_mean(kl_batch)



class ITLRegularizer(Layer):
    """ Identity transform layer that adds cs divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(ITLRegularizer, self).__init__(*args, **kwargs)

    def call(self, inputs, ks, flag, theta):

        X = inputs
        #         print(X)
        if flag == 'sp':
            Z = spherical(K.shape(X))
        elif flag == 'cb':
            Z = clusterdis(K.shape(X))
        elif flag == 'gmm':

            Z = GMM(K.shape(X))
        elif flag == 'sq':
            Z = squre_roll(K.shape(X))
        else:
            Z = swiss_roll(K.shape(X))

        ksize = ks
        Gxx = get_kernel(X, X, ksize)
        Gzz = get_kernel(Z, Z, ksize)
        Gxz = get_kernel(X, Z, ksize)
        #         Gxz = get_kernel(X, Z,ksize)

        r = K.log(K.sqrt(K.mean(Gxx) * K.mean(Gzz) + 1e-5) / (K.mean(Gxz) + 1e-5))
        #         r=K.log(K.sqrt(K.mean(Gxx)*K.mean(Gzz)))-K.log(tf.experimental.numpy.nanmean(Gxz))

        #         print(Gxx,Gzz,Gxz)
        self.add_loss(r * theta, inputs=inputs)

        return inputs, Z, r * theta


class OrthLayer(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(OrthLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, gamma):
        A, B = inputs
        U = K.concatenate((A, B), axis=1)
        #         print(U.shape)
        U = K.l2_normalize(U, axis=1)
        batch = -K.dot(K.transpose(U), U)
        I = tf.eye(K.shape(U)[1])  # tf.Variable(lambda:K.eye(K.shape(U)[1])) #Lambda(lambda t: K.eye(t))()
        batch = (batch + I) ** 2 * gamma
        #         print(batch.shape,I.shape)
        self.add_loss(tf.reduce_mean(batch), inputs=inputs)

        return inputs, tf.reduce_mean(batch)


class MSE_SUP(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(MSE_SUP, self).__init__(*args, **kwargs)

    def call(self, inputs, alpha):
        D, A = inputs
        L = tf.keras.losses.mse(D, A)
        L = tf.reduce_mean(L)

        self.add_loss(L * alpha, inputs=inputs)
        self.add_metric(L* alpha, name="mse_loss",aggregation='mean')
        return inputs, L * alpha


class MSE_UNSUP(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(MSE_UNSUP, self).__init__(*args, **kwargs)

    def call(self, inputs):
        D, A = inputs
        #         X=A * K.log(1e-10 + D) + (1 - A) * K.log(1e-10 + 1 - D)
        #         L = -tf.math.reduce_sum(X)
        #         L=K.mean(K.square(A-D), axis=[1, 2, 3])
        #         L=tf.keras.losses.mse(D,A)
        #         L=Distance(D,A)
        L = tf.keras.losses.mse(D, A)
        L = tf.reduce_mean(L)
        self.add_loss(L * 128 * 128 * 2, inputs=inputs)
        self.add_metric(L* 128 * 128 * 2, name="unmse_loss",aggregation='mean')

        return inputs, L * 128 * 128 * 2


LN2PI = np.log(2 * 3.1415926)




class De_KL(Layer):

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(De_KL, self).__init__(*args, **kwargs)

    def _gaussian_log_density_unsummed(self, z, mu, logvar):
        """First step of Gaussian log-density computation, without summing over dimensions.
        Assumes a diagonal noise covariance matrix.
        """
        diff_sq = (z - mu) ** 2
        inv_var = K.exp(-logvar)
        return - 0.5 * (inv_var * diff_sq + logvar + LN2PI)

    def _gaussian_log_density_unsummed_std_normal(self, z):
        """First step of Gaussian log-density computation, without summing over dimensions.
        Assumes a diagonal noise covariance matrix.
        """
        diff_sq = z ** 2
        return - 0.5 * (diff_sq + LN2PI)

    def _logsumexp(self, x, axis=None, keepdims=False):
        x = tf.convert_to_tensor(x)
        x_max = tf.math.reduce_max(x, axis=axis, keepdims=False)
        temp = tf.math.reduce_sum(K.exp(x - x_max) + 1, axis=axis, keepdims=True)
        ret = K.log(tf.math.reduce_max(temp + 1, tf.zeros_like(temp + 1))) + x_max
        #         np.log(np.max(x, 1e-9))
        if not keepdims:
            ret = tf.math.reduce_sum(ret, axis=axis)
        #         print(K.sum(x,axis=1).numpy())
        return ret  # K.sum(x,axis=1)

    def call(self, inputs, beta):
        z, mu, logvar = inputs
        log_qz_prob = self._gaussian_log_density_unsummed(z[:, None], mu[None, :], logvar[None, :])
        M = tf.nn.relu(K.sum(log_qz_prob, axis=2, keepdims=False))  # ,axis=1,keepdims=False)
        c = tf.nn.relu(log_qz_prob)  # ,axis=1,keepdims=False)
        t1 = K.sum(K.exp(-M) + K.exp(K.sum(log_qz_prob, axis=2, keepdims=False) - M), axis=1, keepdims=False)
        log_qz = K.log(t1 + 1e-5)
        # tf.reduce_logsumexp(K.sum(log_qz_prob, axis=2, keepdims=False),axis=1,keepdims=False)

        # self._logsumexp(K.sum(log_qz_prob, axis=2, keepdims=False),axis=1,keepdims=False)

        #         print(log_qz.shape)

        log_qz_ = tf.linalg.diag(M)  # sum over gaussian dims

        #         print(log_qz_.shape)
        t2 = K.sum(K.exp(-c) + K.exp(log_qz_prob - c), axis=1, keepdims=False)
        log_qz_product = K.sum(  # log_qz_prob,
            #             K.sum(K.log(K.exp(-c)+K.exp(log_qz_prob-c)),axis=1,keepdims=False),
            #             self._logsumexp(log_qz_prob, axis=1,keepdims=False),
            K.log(t2 + 1e-5),
            #             tf.reduce_logsumexp(log_qz_prob, axis=1,keepdims=False),  # logsumexp over batch
            axis=1,  # sum over gaussian dims
            keepdims=False)

        log_pz_prob = self._gaussian_log_density_unsummed_std_normal(z)
        log_pz_product = K.sum(log_pz_prob, axis=1, keepdims=False)  # sum over gaussian dims
        #         print(log_qz_.shape , log_qz.shape)
        idx_code_mi = tf.experimental.numpy.nanmean(log_qz_ - log_qz)
        total_corr = tf.experimental.numpy.nanmean(log_qz - log_qz_product)
        dim_wise_kl = tf.experimental.numpy.nanmean(log_qz_product - log_pz_product)
        idx_code_mi = tf.reduce_mean(idx_code_mi)
        total_corr = tf.reduce_mean(total_corr)
        dim_wise_kl = tf.reduce_mean(dim_wise_kl)
        self.add_loss((idx_code_mi + total_corr * beta + dim_wise_kl + 1e-5), inputs=inputs)

        return inputs, [idx_code_mi, total_corr * beta, dim_wise_kl]
