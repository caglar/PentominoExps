import abc
from abc import ABCMeta
import theano
import theano.tensor as TT

from caglar.core.utils import safe_grad, global_rng, block_gradient, as_floatX, \
        safe_izip, sharedX


class Penalty(object):
    def __init__(self, level=None):
        self.level = level

class ParamPenalty(Penalty):
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def penalize_layer_weights(self, layer):
        pass

    @abc.abstractmethod
    def penalize_params(self, param):
        pass

    @abc.abstractmethod
    def get_penalty_level(self):
        pass


class L2Penalty(Penalty):

    def __init__(self, level=None):
        self.level = level
        self.reg = 0.0

    def penalize_layer_weights(self, layer):
        weight = layer.params.filterby("Weight").values[0]
        self.reg += (weight**2).sum()

    def penalize_params(self, param):
        if isinstance(param, list):
            self.reg += sum((p**2).sum() for p in param)
        else:
            self.reg += (param**2).sum()

    def get_penalty_level(self):
        return self.level * self.reg


class L1Penalty(Penalty):

    def __init__(self, level=None):
        self.level = level
        self.reg = 0.0

    def penalize_layer_weights(self, layer):
        weight = layer.params.filterby("Weight").values[0]
        self.reg += abs(weight).sum()

    def penalize_params(self, param):
        if isinstance(param, list):
            self.reg += sum(abs(p).sum() for p in param)
        else:
            self.reg += abs(param).sum()

    def get_penalty_level(self):
        return self.level * self.reg


class WeightNormConstraint(Penalty):
    """
    Add a norm constraint on the weights of a neural network.
    """
    def __init__(self, limit, min_limit=0, axis=1):
        assert limit is not None, (" Limit for the weight norm constraint should"
                                   " not be empty.")
        self.limit = limit
        self.min_limit = min_limit
        self.max_limit = limit
        self.axis = axis

    def __call__(self, updates, weight_name=None):
        weights = [key for key in updates.keys() if key.name == weight_name]
        if len(weights) != 1:
            raise RuntimeError("More than one weight has been found with "
                               " a name for norm constraint.")

        weight = weights[0]
        updated_W = updates[weight]
        l2_norms = TT.sqrt((updated_W**2).sum(axis=self.axis, keepdims=True))
        desired_norms = TT.clip(l2_norms, self.min_limit, self.max_limit)
        scale = desired_norms / TT.maximum(l2_norms, 1e-7)
        updates[weight] = scale * updated_W
