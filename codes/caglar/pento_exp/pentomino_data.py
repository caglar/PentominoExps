import os
import logging
import numpy as np
from caglar.core.utils.serial import load
from caglar.core.commons import floatX

partition_array = lambda arr, x, y, stride: arr[x * stride:(x + 1) * stride,
                                                y * stride:(y + 1) * stride]

logger = logging.getLogger(__name__)
logger.disabled = False
reshp_img = lambda data: data.reshape(-1, 8, 8, 8, 8).transpose(0, 1, 3,
                                                                2, 4).reshape(-1,
                                                                64, 64)


class PentominoIterator(object):
    """
        The data iterator for the Pentomino.
    """
    def __init__(self,
                 start = None,
                 stop = None,
                 path = None,
                 dir=None, name=None,
                 use_hints=False, mode='train',
                 use_inf_loop=True,
                 conv_X=False,
                 stride=8,
                 batch_size=None):

        self.path = path
        self.dir_ = dir
        if start is None:
            start = 0

        if stop is None:
            stop = self.X.shape[0]

        if mode not in ('train', 'valid', 'test'):
            raise AssertionError

        if stop < start:
            raise AssertionError("Stopping iteration no should be "
                                 " greater than the starting.")

        X, y, hints = self.__load_data()
        self.X = X[start:stop]
        self.y = (np.array(y[start:stop].tolist(), dtype='int32') == 10).astype('uint8')
        self.hints = (np.array(hints[start:stop].tolist(), dtype='int32') + 1).astype('uint8')

        if self.X.shape[0] % batch_size != 0:
            raise AssertionError(" The dataset size should be "
                                 " divisible by the batch size.")

        self.mode = mode
        self.stop = stop
        self.start = start
        self.name = name
        self.batch_size = batch_size
        self.use_hints = use_hints
        self.use_inf_loop = use_inf_loop
        self.img_shape = (64, 64)
        self.npixs = np.prod(self.img_shape)
        self.nclasses = 10
        self.stride = stride
        self.conv_X = conv_X
        self.max_lbl = 10
        self.reset()
        self.data_len = stop - start
        return

    def reset(self):
        self.ex_offset = 0

    def __binarize_labels(self):
        pass

    def __load_data(self):
        abs_path = os.path.join(self.dir_, self.path)
        logger.info('Loading the dataset %s.' % abs_path)
        self.data = load(abs_path)
        return (self.data[0], self.data[1], self.data[2])

    def __iter__(self):
        return self

    def __output_format(self, X, y, hints):
        output = {}
        output['X'] = X.astype(floatX)
        output['y'] = y.flatten().astype('int32')
        output['hints'] = hints.astype('int32')
        return output

    def next(self):
        self.ex_offset += self.batch_size
        X = None
        y = None
        hints = None
        if self.ex_offset + self.batch_size > self.data_len:
            if self.ex_offset < self.data_len:
                X = self.X[self.ex_offset - 1:]
                y = self.y[self.ex_offset - 1:]
                hints = self.hints[self.ex_offset - 1:]
                self.reset()
            elif self.use_inf_loop:
                self.reset()
                ex_idxs = slice(self.ex_offset, self.ex_offset + self.batch_size)
                X = self.X[ex_idxs]
                y = self.y[ex_idxs]
                hints = self.hints[ex_idxs]
            else:
                self.reset()
                raise StopIteration('Iterator has finished.')
        else:
            ex_idxs = slice(self.ex_offset, self.ex_offset + self.batch_size)
            X = self.X[ex_idxs]
            y = self.y[ex_idxs]
            hints = self.hints[ex_idxs]
        if self.conv_X:
            bs = X.shape[0]
            X_dummy = np.zeros((bs,
             8,
             8,
             64))
            for i in xrange(bs):
                cX = X[i].reshape(64, 64)
                for j in xrange(np.prod(self.img_shape[0]) - 1):
                    a = int(j / float(8))
                    b = int(j % float(8))
                    X_dummy[i, a, b] = partition_array(cX, a, b, self.stride).flatten()

            X = X_dummy.reshape((-1, 64, 64))
        return self.__output_format(X, y, hints)
