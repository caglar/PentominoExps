# 2015.08.29 13:52:55 EDT

import os
import logging
import numpy as np
import cPickle as pkl
import theano
from caglar.core.basic import MainLoop
logger = logging.getLogger(__name__)
logger.disabled = False


class PentominoMainLoop(MainLoop):
    """
    PentominoMainLoop
    """

    def __init__(self, model, learning_rate,
                 print_every=None,
                 checkpoint_every=None,
                 inspect_every=None,
                 validate_every=None,
                 train_data_gen=None,
                 train_mon_data_gen=None,
                 valid_data_gen=None,
                 monitor_full_train=False,
                 reload_model=False,
                 max_iters=None,
                 inspect_only=False,
                 valid_iters=None,
                 hints_cost_weight_start=1.0,
                 hints_cost_weight_stop=0.3,
                 hcw_anneal_start=2000,
                 hcw_anneal_rate=0.0001,
                 prefix=None):
        """
            model:
            print_every:
        """
        if prefix is None:
            raise AssertionError('Prefix should not be empty.')

        logger.info('Building the computational graph.')
        super(PentominoMainLoop,
              self).__init__(model=model,
                             learning_rate=learning_rate,
                             checkpoint_every=checkpoint_every,
                             print_every=print_every,
                             inspect_every=inspect_every,
                             inspect_only=inspect_only,
                             monitor_full_train=monitor_full_train,
                             train_data_gen=train_data_gen,
                             train_mon_data_gen=train_mon_data_gen,
                             max_iters=max_iters,
                             valid_data_gen=valid_data_gen,
                             validate_every=validate_every,
                             reload_model=reload_model,
                             prefix=prefix)

        self.hints_cost_weight_start = hints_cost_weight_start
        self.hints_cost_weight_stop = hints_cost_weight_stop
        self.hcw_anneal_start = hcw_anneal_start
        self.hcw_anneal_rate = hcw_anneal_rate
        self.hints_weight = 0
        self.prepare_model()
        return

    def train(self):
        batch = self.train_data_gen.next()
        tdata_x = batch['X']
        tdata_y = batch['y']
        if self.model.use_hints:
            tdata_hints = batch['hints']
            if self.cnt >= self.hcw_anneal_start:
                hcw_diff = abs(
                    self.hints_cost_weight_start - self.hints_cost_weight_stop)
                nsteps = self.cnt - self.hcw_anneal_start
                hcw_delta = self.hcw_anneal_rate * nsteps * hcw_diff
                self.hints_weight = max(
                    self.hints_cost_weight_stop,
                    self.hints_cost_weight_start - hcw_delta)
            elif self.hints_cost_weight_start > 0 and self.hcw_anneal_start > 0:
                self.hints_weight = self.hints_cost_weight_start
            else:
                self.hints_weight = 0.0

        if self.model.use_hints:
            cost, errors, hints_cost, hints_errors, final_cost, gnorm, norm_up, \
                    param_norm, probs = self.train_fn(tdata_x, tdata_y, tdata_hints, self.hints_weight)
        else:
            cost, errors, gnorm, norm_up, param_norm, probs = self.train_fn(tdata_x, tdata_y)

        if self.cnt % self.print_every == 0:
            self.stats['train_cnt'].append(self.cnt)
            self.stats['train_cost'].append(cost)
            self.stats['norm_up'].append(norm_up)
            self.stats['param_norm'].append(param_norm)
            self.stats['errors'].append(errors)

            if self.model.use_hints:
                self.stats['train_hints_cost'].append(hints_cost)
                self.stats['train_hints_error'].append(hints_errors)
                self.stats['train_hints_final_cost'].append(final_cost)
                train_str = (" Iter %d: cost: %f, hint cost: %f, update norm: %f parameter norm: "
                             " %f,  norm of the gradients: %f,  errors: %f,  hints errors: %f "
                             " final cost: %f hints weight: %f ")
                train_str_vals = (self.cnt, cost, hints_cost, norm_up,
                                  param_norm, gnorm, errors, hints_errors,
                                  final_cost, self.hints_weight)
            else:
                train_str = (" Iter %d: cost: %f, update norm: %f parameter norm: %f, "
                            " norm of the gradients: %f  errors: %f ")

                train_str_vals = (self.cnt, cost, norm_up, param_norm, gnorm,
                                  errors)

            logger.info(train_str % train_str_vals)

    def validate(self, data_gen=None, mode='Valid'):
        if mode == 'Valid':
            logger.info('Validating the model...')
        else:
            logger.info('Evaluating the model on %s dataset...' % mode)

        costs = []
        errors = []
        costs_hints = []
        errors_hints = []
        final_cost = 0

        if data_gen is None:
            data_gen = self.valid_data_gen

        if self.valid_iters:
            for i in xrange(self.valid_iters):
                batch = data_gen.next()
                try:
                    batch = data_gen.next()
                    vdata_x, vdata_y, vdata_hints = batch['X'], batch[
                        'y'
                    ], batch['hints']
                except:
                    data_gen.reset()
                    break
                if self.model.use_hints:
                    cost, cost_hints, error, error_hints = self.valid_fn(
                        vdata_x, vdata_y, vdata_hints)
                else:
                    cost, error = self.valid_fn(vdata_x, vdata_y)
                costs.append(cost)
                errors.append(error)
                if self.model.use_hints:
                    costs_hints.append(cost_hints)
                    errors_hints.append(error_hints)

        else:
            for batch in data_gen:
                if self.model.use_hints:
                    vdata_x, vdata_y, vdata_hints = batch['X'], batch['y'], \
                            batch['hints']
                else:
                    vdata_x, vdata_y = batch['X'], batch['y']
                if self.model.use_hints:
                    cost, error, cost_hints, error_hints, final_cost, probs = \
                            self.valid_fn(vdata_x, vdata_y, vdata_hints, self.hints_weight)
                else:
                    cost, error, probs = self.valid_fn(vdata_x, vdata_y)

                costs.append(cost)
                errors.append(error)

                if self.model.use_hints:
                    costs_hints.append(cost_hints)
                    errors_hints.append(error_hints)

        error = np.mean(errors)
        cost = np.mean(costs)

        if self.model.use_hints:
            final_cost = final_cost.tolist()
            hints_error = np.mean(errors_hints)
            hints_cost = np.mean(costs_hints)

        if self.model.use_hints:
            valid_str_errors = (" %(mode)s error: %(error)f, "
                                " %(mode)s hints error: %(hints_error)f, "
                                " %(mode)s cost %(cost)f, %(mode)s hints cost: %(hints_cost)f, "
                                " %(mode)s final cost: %(final_cost)f ") % locals()
        else:
            valid_str_errors = '%(mode)s error: %(error)s, %(mode)s cost %(cost)f ' % locals()

        self.stats[mode + '_full_errors'].append(error)
        self.stats[mode + '_full_costs'].append(cost)

        if self.model.use_hints:
            self.stats[mode + '_full_errors_hints'].append(hints_error)
            self.stats[mode + '_full_costs_hints'].append(hints_cost)
            self.stats[mode + '_full_final_costs'].append(final_cost)

        logger.info(valid_str_errors)
        if mode == 'Valid':
            self.model.params.print_param_norms()

        if abs(cost) <= self.best_cost or error <= self.best_error:
            logger.info('Saving the best model.')
            self.best_cost = abs(cost)
            self.best_error = abs(error)
            self.save(mdl_name=self.best_mdl_name,
                      stats_name=self.best_stats_name)

    def inspect_model(self, data_x, data_y, data_hints=None):
        if not (self.model.use_hints and data_hints is None):
            raise AssertionError("When the model is specified to use hints. Please"
                                 "provide the hints from the generator to the inspect_fn.")
        raise NotImplementedError('Inspection function has not been implemented yet.')

