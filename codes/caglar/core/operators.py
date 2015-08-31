import numpy as np

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import scipy
import scipy.linalg
from scipy.linalg import circulant

from caglar.core.commons import EPS, Sigmoid, Tanh, floatX
from caglar.core.utils import concatenate, get_key_byname_from_dict, sharedX, \
                                as_floatX, block_gradient


class Operator(object):
    def __init__(self, eps=1e-8):
        if eps is None:
            self.eps = EPS
        else:
            self.eps = eps

class Dropout(Operator):
    """
        Perform the dropout on the layer.
    """
    def __init__(self, dropout_prob=0.5, rng=None):
        self.rng = RandomStreams(1) if rng is None else rng
        self.dropout_prob = dropout_prob

    def __call__(self, input, deterministic=False):

       if input is None:
            raise ValueError("input for the %s should not be empty." % __class__.__name__)
       p = self.dropout_prob
       if deterministic:
           return input
       else:
           retain_p = 1 - p
           input /= retain_p
           return input * self.rng.binomial(input.shape,
                                            p=retain_p,
                                            dtype=floatX)


class GaussianNoise(Operator):

    def __init__(self, avg=0, std=0.01, rng=None):
        self.rng = RandomStreams(1) if rng is None else rng
        self.avg = avg
        self.std = std

    def __call__(self):
        raise NotImplementedError("call function is not implemented!")


class AdditiveGaussianNoise(GaussianNoise):
    """
        Perform the dropout on the layer.
    """
    def __call__(self, input, deterministic=False):

       if input is None:
            raise ValueError("input for the %s should not be empty." % __class__.__name__)
       p = self.dropout_prob
       if deterministic:
           return input
       else:
           return input + self.rng.normal(input.shape,
                                          avg = self.avg,
                                          std = self.std,
                                          dtype=floatX)


class MultiplicativeGaussianNoise(GaussianNoise):
    """
        Perform the dropout on the layer.
    """
    def __call__(self, input, deterministic=False):

       if input is None:
            raise ValueError("input for the %s should not be empty." % __class__.__name__)
       if deterministic:
           return input
       else:
           return input * self.rng.normal(input.shape,
                                          avg = self.avg,
                                          std = self.std,
                                          dtype=floatX)
       return result


