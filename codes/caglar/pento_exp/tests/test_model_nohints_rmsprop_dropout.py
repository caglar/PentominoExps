import theano
import cPickle as pkl
import warnings

import numpy as np

from caglar.core.parameters import (WeightInitializer,
                                    InitMethods,
                                    BiasInitMethods,
                                    BiasInitializer)

from caglar.pento_exp.pmodel import PentoModel
from caglar.core.learning_rule import Adasecant, Adam, RMSPropMomentum, Adasecant2

import theano
import theano.tensor as TT
from caglar.core.commons import Leaky_Rect, Sigmoid, Rect, Tanh, Softmax, Linear
from caglar.pento_exp.mainloop import PentominoMainLoop
from caglar.pento_exp.pentomino_data import PentominoIterator
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from caglar.core.nan_guard import NanGuardMode


start = 0
stop = 10000
tstop = 80000
path = "/data/lisa/data/pentomino/datasets/"
tsdir_ = "pento64x64_40k_64patches_seed_975168712_64patches.npy"
trdir_ = "pento64x64_80k_64patches_seed_735128712_64patches.npy"
seed = 7

use_hints = False
batch_size = 400

rng = np.random.RandomState(seed)
trng = RandomStreams(seed)

reshp_img = lambda data: data.reshape(batch_size,
                                      8, 8,
                                      8, 8).transpose(0, 1,
                                      3, 2,
                                      4).reshape(batch_size,
                                                 64, 64)


train_data_iter = PentominoIterator(start=start,
                                    stop=tstop, path=path,
                                    dir=trdir_, name="train",
                                    use_inf_loop=True,
                                    use_hints=use_hints,
                                    conv_X=True,
                                    batch_size=batch_size)

train_mon_data_iter = PentominoIterator(start=start,
                                        stop=tstop, path=path,
                                        dir=trdir_, name="train",
                                        use_inf_loop=False,
                                        use_hints=use_hints,
                                        conv_X=True,
                                        batch_size=batch_size)


valid_data_iter = PentominoIterator(start=start,
                                    stop=stop,
                                    path=path,
                                    dir=tsdir_, name="valid",
                                    use_inf_loop=False,
                                    use_hints=use_hints,
                                    conv_X=True,
                                    batch_size=batch_size)

"""
learning_rule = Adasecant2(delta_clip=25,
                           use_adagrad=True,
                           grad_clip=0.5,
                           gamma_clip=0.0)


#learning_rule = Adam(gradient_clipping=5.0)
#

"""
learning_rule = RMSPropMomentum(init_momentum=0.92)
#"""

def get_inps(use_hints=True, vgen=None, debug=False):
    if use_hints:
        X, y, hints = TT.ftensor3("X"), TT.ivector("y"), TT.imatrix("hints")
        if debug:
            theano.config.compute_test_value = "warn"
            batch = vgen.next()
            X.tag.test_value = batch['X'].astype("float32")
            y.tag.test_value = batch['y'].astype("int32")
            hints.tag.test_value = batch['hints'].astype("int32")
        return [X, y, hints]
    else:
        X, y = TT.ftensor3("X"), TT.ivector("y")
        if debug:
            theano.config.compute_test_value = "warn"
            batch = vgen.next()
            X.tag.test_value = batch['x']
            y.tag.test_value = batch['y'].astype("int32")
        return [X, y]


inps = get_inps(use_hints=use_hints, vgen=valid_data_iter, debug=False)

n_in = 64*64
n_hids = 512
n_bottleneck = 11
learning_rate = 4e-4

std = 0.05

wi = WeightInitializer(sparsity=-1,
                       scale=std,
                       rng=rng,
                       init_method=InitMethods.Adaptive,
                       center=0.0)

bi = BiasInitializer(sparsity=-1,
                     scale=std,
                     rng=rng,
                     init_method=BiasInitMethods.Constant,
                     center=0.0)

mode = None

pmodel = PentoModel(n_in=n_in,
                    n_hids=n_hids,
                    n_bottleneck=n_bottleneck,
                    learning_rule=learning_rule,
                    n_out=2,
                    activ=Rect,
                    weight_norm_constraint=None,
                    inps=inps,
                    bottleneck_activ=Linear,
                    theano_function_mode=mode,
                    patch_size=64,
                    use_deepmind_pooling=False,
                    weight_initializer=wi,
                    dropout=0.5,
                    bias_initializer=bi,
                    use_hints=use_hints,
                    batch_size=batch_size,
                    name="pento_model2_nohints_rmsprop")

mainloop = PentominoMainLoop(pmodel,
                             learning_rate=learning_rate,
                             print_every=20,
                             checkpoint_every=320,
                             inspect_every=None,
                             validate_every=300,
                             valid_data_gen=valid_data_iter,
                             train_data_gen=train_data_iter,
                             train_mon_data_gen=train_mon_data_iter,
                             monitor_full_train=True,
                             hints_cost_weight_stop=0.0,
                             hints_cost_weight_start=0.0,
                             hcw_anneal_start=0,
                             reload_model=False,
                             max_iters=80000,
                             prefix="pento_mainloop_nohints2_rmsprop")

mainloop.run()
