import random
import numpy as np

def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    # print "shape = ", shape
    shape[0] = batch_size
    batch_s = np.empty(shape)   # Return a new array of uninitialized (arbitrary) data of the given shape, dtype, and order.

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    # print "batch_s = ", batch_s
    # print "_train = ", _train[:batch_size]
    return batch_s


def one_hot(y_, class_num=40):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    # print "y_.shape = ", y_.shape
    # y_ = y_.reshape(len(y_))
    # print "np.max(y_) = ", np.max(y_)
    # n_values = int(np.max(y_)) + 1
    # print np.eye(n_values)[np.array(y_, dtype=np.int32)].shape


    return np.eye(class_num)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


class DataReader(object):
    def __init__(self, _input, _output, _batch_size, _input_size, _num_unroll_steps):
        self.input = _input
        self.output = _output
        self.batch_size = _batch_size
        self.input_size = _input_size
        self.num_unroll_steps = _num_unroll_steps
        self.length = _input.shape[0]
        self.class_num = _output.shape[1]

        # round down length to whole number of slices
        reduced_length = (self.length // (self.batch_size * self.num_unroll_steps)) * self.batch_size * self.num_unroll_steps

        print "class_num = %s \t input length = %s \t reduced_length = %s" % (self.class_num, self.length, reduced_length)
        '''
        - self.class_num = 40
        - self.input length = 125973
        - reduced_length = 125952
        '''

        _input  = self.input[:reduced_length, :]
        _output = self.output[:reduced_length, :]
        print '_input.shape = %s \t _output.shape = %s'% (_input.shape, _output.shape)
        x_batches = _input.reshape([-1, self.batch_size, self.num_unroll_steps, self.input_size])
        y_batches = _output.reshape([-1, self.batch_size, self.class_num])
        print 'x_batches.shape = %s \t y_batches.shape = %s'% (x_batches.shape, y_batches.shape)
        '''Training set & Testing set
        - _input.shape = (125952, 122)
        - _output.shape = (125952, 40)
        - x_batches.shape = (128, 984, 122)
        - y_batches.shape = (128, 984, 40)
        '''

        # x_batches = np.transpose(x_batches, axes=(1, 0, 2, 3))
        # y_batches = np.transpose(y_batches, axes=(1, 0, 2))
        print '-'*50
        print "Format : (batch_group, batch_size, dim(features or labels)"
        print 'x_batches.shape = %s \t y_batches.shape = %s'% (x_batches.shape, y_batches.shape)

        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)
        assert len(self._x_batches) == len(self._y_batches)
        self.length = len(self._y_batches)

    def shuffle_arr_list(self):
        zip_ = list(zip(self._x_batches, self._y_batches))
        random.shuffle(zip_)
        x_batches, y_batches = zip(*zip_)
        self._x_batches = list(x_batches)
        self._y_batches = list(y_batches)

    def iter_batches(self):
        for x, y in zip(self._x_batches, self._y_batches):
                yield x, y
