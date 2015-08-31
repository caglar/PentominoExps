"""
Main structure and most of the codes for the learning rules here are
borrowed from pylearn2 with some small modifications:

https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/training_algorithms/learning_rule.py

"""

import numpy as np
from theano import config, ifelse
from theano import tensor as T
import logging

from theano.compat.python2x import OrderedDict
from caglar.core.utils import sharedX, as_floatX

logger = logging.getLogger(__name__)


class LearningRule():

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        Method called by the training algorithm, which allows LearningRules to
        add monitoring channels.

        Parameters
        ----------
        monitor : pylearn2.monitor.Monitor
            Monitor object, to which the rule should register additional
            monitoring channels.
        monitoring_dataset : pylearn2.datasets.dataset.Dataset or dict
            Dataset instance or dictionary whose values are Dataset objects.
        """
        pass

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Provides the symbolic (theano) description of the updates needed to
        perform this learning rule.

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.

        Returns
        -------
        updates : OrderdDict
            A dictionary mapping from the old model parameters, to their new
            values after a single iteration of the learning rule.

        Notes
        -----
        e.g. for standard SGD, one would return `sgd_rule_updates` defined
        below. Note that such a `LearningRule` object is not implemented, as
        these updates are implemented by default when the `learning_rule`
        parameter of sgd.SGD.__init__ is None.

        .. code-block::  python

            sgd_rule_updates = OrderedDict()
            for (param, grad) in grads.iteritems():
                sgd_rule_updates[k] = (param - learning_rate *
                                       lr_scalers.get(param, 1.) * grad)
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                                  "get_updates.")


class Momentum(LearningRule):
    """
    Implements momentum as described in Section 9 of
    "A Practical Guide to Training Restricted Boltzmann Machines",
    Geoffrey Hinton.

    Parameters are updated by the formula:
    inc := momentum * inc - learning_rate * d cost / d param
    param := param + inc

    Parameters
    ----------
    init_momentum : float
        Initial value for the momentum coefficient. It remains fixed during
        training unless used with a `MomentumAdjustor`
        extension.
    nesterov_momentum: bool
        Use the accelerated momentum technique described in:
        "Advances in Optimizing Recurrent Networks", Yoshua Bengio, et al.

    """

    def __init__(self, init_momentum, nesterov_momentum=False):
        assert init_momentum >= 0.
        assert init_momentum < 1.
        self.momentum = sharedX(init_momentum, 'momentum')
        self.nesterov_momentum = nesterov_momentum

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        Activates monitoring of the momentum.

        Parameters
        ----------
        monitor : pylearn2.monitor.Monitor
            Monitor object, to which the rule should register additional
            monitoring channels.
        monitoring_dataset : pylearn2.datasets.dataset.Dataset or dict
            Dataset instance or dictionary whose values are Dataset objects.
        """
        monitor.add_channel(
            name='momentum',
            ipt=None,
            val=self.momentum,
            data_specs=(NullSpace(), ''),
            dataset=monitoring_dataset)

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Provides the updates for learning with gradient descent + momentum.

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """

        updates = OrderedDict()

        for (param, grad) in six.iteritems(grads):
            vel = sharedX(param.get_value() * 0.)
            assert param.dtype == vel.dtype
            assert grad.dtype == param.dtype
            if param.name is not None:
                vel.name = 'vel_' + param.name

            scaled_lr = learning_rate * lr_scalers.get(param, 1.)
            updates[vel] = self.momentum * vel - scaled_lr * grad

            inc = updates[vel]
            if self.nesterov_momentum:
                inc = self.momentum * inc - scaled_lr * grad

            assert inc.dtype == vel.dtype
            updates[param] = param + inc

        return updates


class RMSPropMomentum(Momentum):
    """
    Implements the RMSprop

    Parameters
    ----------
    """

    def __init__(self,
                 init_momentum,
                 averaging_coeff=0.95,
                 stabilizer=1e-2,
                 use_first_order=False,
                 bound_inc=False,
                 momentum_clipping=None):
        init_momentum = float(init_momentum)
        assert init_momentum >= 0.
        assert init_momentum <= 1.
        averaging_coeff = float(averaging_coeff)
        assert averaging_coeff >= 0.
        assert averaging_coeff <= 1.
        stabilizer = float(stabilizer)
        assert stabilizer >= 0.

        self.__dict__.update(locals())
        del self.self
        self.momentum = sharedX(self.init_momentum)

        self.momentum_clipping = momentum_clipping
        if momentum_clipping is not None:
            self.momentum_clipping = np.cast[config.floatX](momentum_clipping)

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        .. todo::

            WRITEME
        """
        updates = OrderedDict()
        velocity = OrderedDict()
        tot_norm_up = 0
        tot_param_norm = 0

        for param in grads.keys():

            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))
            velocity[param] = sharedX(np.zeros_like(param.get_value()))

            if param.name is not None:
                avg_grad_sqr.name = 'avg_grad_sqr_' + param.name

            new_avg_grad_sqr = self.averaging_coeff * avg_grad_sqr +\
                (1 - self.averaging_coeff) * T.sqr(grads[param])
            if self.use_first_order:
                avg_grad = sharedX(np.zeros_like(param.get_value()))
                if param.name is not None:
                    avg_grad.name = 'avg_grad_' + param.name
                new_avg_grad = self.averaging_coeff * avg_grad +\
                    (1 - self.averaging_coeff) * grads[param]
                rms_grad_t = T.sqrt(new_avg_grad_sqr - new_avg_grad**2)
                updates[avg_grad] = new_avg_grad
            else:
                rms_grad_t = T.sqrt(new_avg_grad_sqr)
            rms_grad_t = T.maximum(rms_grad_t, self.stabilizer)
            normalized_grad = grads[param] / (rms_grad_t)
            new_velocity = self.momentum * velocity[param] -\
                learning_rate * normalized_grad
            tot_norm_up += new_velocity.norm(2)
            tot_param_norm += param.norm(2)


            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[velocity[param]] = new_velocity
            updates[param] = param + new_velocity

        if self.momentum_clipping is not None:
            tot_norm_up = 0


            new_mom_norm = sum(
                map(lambda X: T.sqr(X).sum(),
                    [updates[velocity[param]] for param in grads.keys()])
            )
            new_mom_norm = T.sqrt(new_mom_norm)
            scaling_den = T.maximum(self.momentum_clipping, new_mom_norm)
            scaling_num = self.momentum_clipping

            for param in grads.keys():
                if self.bound_inc:
                    updates[velocity[param]] *= (scaling_num / scaling_den)
                    updates[param] = param + updates[velocity[param]]
                else:
                    update_step = updates[velocity[param]] *(scaling_num / scaling_den)
                    tot_norm_up += update_step.norm(2)
                    updates[param] = param + update_step

        return updates, tot_norm_up, tot_param_norm


class Adam(Momentum):
    """
    The MIT License (MIT)

    Copyright (c) 2015 Alec Radford

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    This is adapted from the code written by, Junyoung Chung and Laurent Dinh.

    """
    def __init__(self,
                 init_momentum=0.9,
                 averaging_coeff=0.99,
                 stabilizer=1e-4,
                 gradient_clipping=None):
        init_momentum = float(init_momentum)
        assert init_momentum >= 0.
        assert init_momentum <= 1.
        averaging_coeff = float(averaging_coeff)
        assert averaging_coeff >= 0.
        assert averaging_coeff <= 1.
        stabilizer = float(stabilizer)
        assert stabilizer >= 0.

        self.__dict__.update(locals())
        del self.self
        self.momentum = sharedX(self.init_momentum)

        self.gradient_clipping = gradient_clipping
        if gradient_clipping is not None:
            self.gradient_clipping = np.cast[config.floatX](gradient_clipping)

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        .. todo::

            WRITEME
        """
        updates = OrderedDict()
        velocity = OrderedDict()
        counter = sharedX(0, 'counter')
        tot_norm_up = 0
        tot_param_norm = 0

        if self.gradient_clipping is not None:
            grads_norm = sum(
                map(lambda X: T.sqr(X).sum(),
                    [grads[param] for param in grads.keys()])
            )
            grads_norm = T.sqrt(grads_norm)
            scaling_den = T.maximum(self.gradient_clipping, grads_norm)
            scaling_num = self.gradient_clipping

            for param in grads.keys():
                grads[param] = scaling_num * grads[param] / scaling_den

        for param in grads.keys():

            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))
            velocity[param] = sharedX(np.zeros_like(param.get_value()))

            next_counter = counter + 1.

            fix_first_moment = 1. - self.momentum**next_counter
            fix_second_moment = 1. - self.averaging_coeff**next_counter

            if param.name is not None:
                avg_grad_sqr.name = 'avg_grad_sqr_' + param.name

            new_avg_grad_sqr = self.averaging_coeff*avg_grad_sqr \
                + (1 - self.averaging_coeff)*T.sqr(grads[param])

            rms_grad_t = T.sqrt(new_avg_grad_sqr)
            rms_grad_t = T.maximum(rms_grad_t, self.stabilizer)
            new_velocity = self.momentum * velocity[param] \
                - (1 - self.momentum) * grads[param]

            normalized_velocity = (new_velocity * T.sqrt(fix_second_moment)) \
                / (rms_grad_t * fix_first_moment)
            tot_param_norm += param.norm(2)
            tot_norm_up += learning_rate*normalized_velocity.norm(2)
            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[velocity[param]] = new_velocity
            updates[param] = param + learning_rate * normalized_velocity

        updates[counter] = counter + 1

        return updates, tot_norm_up, tot_param_norm


class AdaDelta(LearningRule):
    """
    Implements the AdaDelta learning rule as described in:
    "AdaDelta: An Adaptive Learning Rate Method", Matthew D. Zeiler.

    Parameters
    ----------
    decay : float, optional
        Decay rate :math:`\\rho` in Algorithm 1 of the aforementioned
        paper.
    """

    def __init__(self, decay=0.95):
        assert decay >= 0.
        assert decay < 1.
        self.decay = decay

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Compute the AdaDelta updates

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """
        tot_norm_up = 0
        tot_param_norm = 0

        updates = OrderedDict()
        for param in grads.keys():

            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(param.get_value() * 0.)
            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = sharedX(param.get_value() * 0.)

            if param.name is not None:
                mean_square_grad.name = 'mean_square_grad_' + param.name
                mean_square_dx.name = 'mean_square_dx_' + param.name

            # Accumulate gradient
            new_mean_squared_grad = (
                self.decay * mean_square_grad +
                (1 - self.decay) * T.sqr(grads[param])
            )

            # Compute update
            epsilon = 1e-6#lr_scalers.get(param, 1.) * learning_rate
            rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
            delta_x_t = - rms_dx_tm1 / rms_grad_t * grads[param]

            # Accumulate updates
            new_mean_square_dx = (
                self.decay * mean_square_dx +
                (1 - self.decay) * T.sqr(delta_x_t)
            )

            tot_param_norm += param.norm(2)
            tot_norm_up += delta_x_t.norm(2)

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[param] = param + delta_x_t

        return updates, tot_norm_up, tot_param_norm

