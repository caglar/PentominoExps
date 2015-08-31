import logging
from collections import OrderedDict
import cPickle as pkl
import warnings

import theano
import theano.tensor as TT
from caglar.core.parameters import (WeightInitializer,
                                    BiasInitializer)

from caglar.core.layers import (AffineLayer,
                                ConvMLPLayer)

from caglar.core.parameters import Parameters
from caglar.core.basic import Model
from caglar.core.costs import kl, nll, huber_loss, nll_hints
from caglar.core.utils import safe_grad, global_rng, block_gradient, as_floatX, \
        safe_izip, sharedX

from caglar.core.operators import Dropout

from caglar.core.commons import Leaky_Rect, Sigmoid, Rect, Softmax, Tanh, Linear
from caglar.core.timer import Timer
from caglar.core.penalty import L2Penalty, WeightNormConstraint


logger = logging.getLogger(__name__)
logger.disabled = False


class PentoModel(Model):
    """
        NTM model.
    """
    def __init__(self,
                 n_in,
                 n_hids,
                 n_bottleneck,
                 n_out,
                 activ=None,
                 inps=None,
                 bottleneck_activ=None,
                 patch_size=64,
                 weight_initializer=None,
                 use_deepmind_pooling=False,
                 theano_function_mode=None,
                 monitor_gnorm=False,
                 use_grad_norms=True,
                 use_hints=False,
                 dropout = None,
                 bias_initializer=None,
                 weight_norm_constraint=None,
                 learning_rule=None,
                 batch_size=128,
                 npatches=64,
                 l2_reg=None,
                 l1_reg=None,
                 name=None):

        self.n_in = n_in
        self.n_hids = n_hids
        self.n_bottleneck = 11
        self.npatches = npatches
        self.use_deepmind_pooling = use_deepmind_pooling
        self.weight_norm_constraint = weight_norm_constraint

        self.bottleneck_out = self.npatches if self.use_deepmind_pooling \
                else self.npatches * self.n_bottleneck

        logger.info("The bottleneck output being used is %d" % self.bottleneck_out)

        self.theano_function_mode = theano_function_mode
        self.monitor_gnorm = monitor_gnorm

        self.dropOp = None
        if dropout:
            logger.info("Dropout is enabled.")
            self.dropOp = Dropout(dropout_prob=dropout)

        self.n_out = n_out
        self.activ = activ
        self.learning_rule = learning_rule
        self.use_hints = use_hints
        self.use_grad_norms = use_grad_norms
        self.bottleneck_activ = bottleneck_activ if not use_deepmind_pooling else Softmax
        self.patch_size = patch_size
        self.npatches = npatches
        self.inps = inps
        self.batch_size = batch_size
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.name = name
        self.reset()
        super(PentoModel, self).__init__()

    def reset(self):
        self.X = self.inps[0]
        self.y = self.inps[1]
        self.hints = None
        self.hints_weight = None
        if self.use_hints:
            self.hints = self.inps[2]
            self.hints_weight = TT.fscalar("hints_weight")

        self.params = Parameters()
        self.conv_mlp_layer = None
        self.out_layer = None
        self.out_hid_layer = None
        self.updates = OrderedDict({})
        self.probs = None
        self.hid_rep = None
        self.reg = 0

    def collect_grad_stats(self, grads):
        gnorm_d = OrderedDict({})
        for k, v in grads:
            gnorm = sharedX(0., name="grad_%s" % k)
            gnorm_d[gnorm] = ((v**2).sum())**0.5
        return gnorm_d

    def build_model(self):
        if not self.conv_mlp_layer:
            self.conv_mlp_layer = ConvMLPLayer(patch_size=self.patch_size,
                                               npatches=self.npatches,
                                               n_hids=self.n_hids,
                                               activ=self.activ,
                                               n_out=self.n_bottleneck,
                                               out_activ=self.bottleneck_activ,
                                               use_deepmind_pooling=self.use_deepmind_pooling,
                                               weight_initializer=self.weight_initializer,
                                               bias_initializer=self.bias_initializer,
                                               name=self.pname("conv_mlp_layer"))

        if not self.out_hid_layer:
            self.out_hid_layer = AffineLayer(n_in=self.bottleneck_out,
                                            n_out=self.n_hids,
                                            use_bias=True,
                                            weight_initializer=self.weight_initializer,
                                            bias_initializer=self.bias_initializer,
                                            name=self.pname("out_hid_layer"))

        if not self.out_layer:
            self.out_layer = AffineLayer(n_in=self.n_hids,
                                         n_out=self.n_out,
                                         use_bias=True,
                                         weight_initializer=self.weight_initializer,
                                         bias_initializer=self.bias_initializer,
                                         name=self.pname("out_layer"))

        self.children = [self.conv_mlp_layer, self.out_hid_layer, self.out_layer]
        self.merge_params()
        self.str_params()

    def standardize_layer(self, inp):
        mean = inp.mean(axis=0, keepdims=True)
        std = inp.std(axis=0, keepdims=True)
        res = (inp - mean) / TT.maximum(std, 1e-7)
        return res

    def get_cost(self, inp, target, hints=None, deterministic=False):
        if self.use_hints:
            if hints is None:
                if self.hints is None:
                    raise RuntimeError("Hints should be empty.")
                else:
                    hints = self.hints

        if not self.probs or not self.hid_rep:
            self.probs, self.hid_rep = self.fprop(inp, deterministic=deterministic)

        if self.use_hints:
            if not isinstance(self.hid_rep, TT.nnet.SoftmaxWithBias):
                hid_rep = Softmax(self.hid_rep)
            else:
                hid_rep = self.hid_rep

            self.cost_hints, self.hints_errors = nll_hints(hints, Softmax(self.hid_rep))

        self.cost, self.errors = nll(target, self.probs)

        if self.cost.ndim > 1:
            self.cost = self.cost.mean()
        else:
            self.cost /= target.shape[0]

        return self.cost, self.errors

    def get_inspect_fn(self):
        pass

    def get_train_fn(self, lr=None, mdl_name=None):
        logger.info("Compiling the training function.")

        if lr is None:
            lr = self.eps

        cost, errors = self.get_cost(inp=self.X,
                                     target=self.y,
                                     hints=self.hints)

        final_cost = cost
        outs = [cost, errors]
        params = self.params.values

        if self.use_hints:
            final_cost = self.hints_weight * self.cost_hints + \
                    (1 - self.hints_weight) * cost
            outs += [self.cost_hints, self.hints_errors, final_cost]

        grads = safe_grad(final_cost, params)
        gnorm = sum(grad.norm(2) for _, grad in grads.iteritems())
        updates, norm_up, param_norm = self.learning_rule.get_updates(learning_rate=lr,
                                                                      grads=grads)
        if self.weight_norm_constraint:
            logger.info("Adding the weight norm constraint")
            paramname = self.out_hid_layer.params.getparamname("weight")
            normconstraint = WeightNormConstraint(self.weight_norm_constraint, axis=1)
            normconstraint(updates, paramname)

        outs += [gnorm, norm_up, param_norm]
        if self.use_hints:
            inps = self.inps +  [self.hints_weight]
        else:
            inps = self.inps

        train_fn = theano.function(inps,
                                   outs + [self.probs],
                                   updates=updates,
                                   mode=self.theano_function_mode,
                                   name = self.pname("train_fn"))

        return train_fn

    def get_valid_fn(self, mdl_name=None):
        logger.info("Compiling the validation function.")

        if self.use_hints:
            cost, errors = self.get_cost(inp=self.X, target=self.y,
                                         hints=self.hints,
                                         deterministic=True)
        else:
            cost, errors = self.get_cost(inp=self.X, target=self.y,
                                         deterministic=True)

        final_cost = cost
        outs = [cost, errors]

        if self.use_hints:
            final_cost = self.hints_weight * self.cost_hints + (1 - self.hints_weight) * cost
            outs += [self.cost_hints, self.hints_errors, final_cost]

        if self.use_hints:
            inps = self.inps + [self.hints_weight]
        else:
            inps = self.inps
        valid_fn = theano.function(inps,
                                   outs + [self.probs],
                                   mode=self.theano_function_mode,
                                   name=self.pname("valid_fn"))

        return valid_fn

    def fprop(self,
              inp=None,
              deterministic=False):

        if inp is None:
            self.X = inp

        if not self.use_hints:
            self.hints = None

        self.build_model()
        hid_rep = self.conv_mlp_layer.fprop(self.X)

        if False:
            out_hid_in = self.standardize_layer(Softmax(hid_rep.reshape((self.batch_size,
                                                                     -1))))
        else:
            out_hid_in = Rect(hid_rep.reshape((self.batch_size, -1)))

        if self.dropOp:
            hid_rep = self.dropOp(hid_rep, deterministic=deterministic)

        out_hid_layer = Rect(self.out_hid_layer.fprop(out_hid_in))

        if self.dropOp:
            hid_rep = self.dropOp(out_hid_layer, deterministic=deterministic)

        self.probs = Softmax(self.out_layer.fprop(out_hid_layer))
        return self.probs, hid_rep

