import tensorflow as tf

class LSTM(object):
    def __init__(self,  _batch_size, _learning_rate, **kwargs):
        self.learning_rate = _learning_rate
        self.batch_size = _batch_size
        self.num_class = kwargs.pop('num_class', 5)
        self.dropout = kwargs.pop('dropout', 0.0)
        self.hidden_size = kwargs.pop('hidden_size', 32)
        #self.input_size = kwargs.pop('input_size', 64)
        self.num_input = kwargs.pop('num_input', 122)
        # self.batch_size = kwargs.pop('batch_size', 20)
        self.num_rnn_layers = kwargs.pop('num_rnn_layers', 2)
        self.num_unroll_steps = kwargs.pop('num_unroll_steps', 14)

        self.batch_size = kwargs.pop('batch_size', 32)
        #self.input_ = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_unroll_steps, self.input_size])  # input_size = dim(features)
        self.input_ = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_unroll_steps, self.num_input])  # input_size = dim(features)
        self.output_ = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class])

        # Graph weights
        self._weights = {
            'hidden': tf.Variable(tf.random_normal([self.num_input, self.hidden_size])), # Hidden layer weights
            'out': tf.Variable(tf.random_normal([self.hidden_size, self.num_class], mean=1.0))
        }
        self._biases = {
            'hidden': tf.Variable(tf.random_normal([self.hidden_size])),
            'out': tf.Variable(tf.random_normal([self.num_class]))
        }

    # character-level lstm-cnn
    def create_rnn_cell(self):
        '''tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias, state_is_tuple)
        - num_units : The number of units in the LSTM cell = hidden_size
        - state_is_tuple=True : accepted and returned states are 2-tuples of the "c_state" and "m_state" = (c_state, m_state)
        '''
        # cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True, forget_bias=1.0)
        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        if self.dropout > 0.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.-self.dropout)
        return cell

    def LSTM_RNN(self):
        # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
        # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
        # Note, some code of this notebook is inspired from an slightly different
        # RNN architecture used on another dataset, some of the credits goes to
        # "aymericdamien" under the MIT license.

        # # (NOTE: This step could be greatly optimised by shaping the dataset once
        # # input shape: (batch_size, timestep_size, input_size)
        # _input = tf.transpose(self._input, [1, 0, 2])  # permute num_unroll_steps and batch_size
        # # Reshape to prepare input to hidden activation
        # _input = tf.reshape(_input, [-1, self.num_input])

        # input shape: (batch_size, input_size)

        # _input = tf.reshape(self._input, [-1, self.num_unroll_steps, self.num_input]
        _input = tf.reshape(self.input_, [-1, self.num_unroll_steps, self.num_input, 1])

        conv1_1 = tf.layers.conv2d(_input,  64, kernel_size=[3, 3], strides=[1, 1], activation=tf.nn.relu, padding='same', name="conv1_1")
        conv1_2 = tf.layers.conv2d(conv1_1, 32, kernel_size=[3, 3], strides=[1, 1], activation=tf.nn.relu, padding='same', name="conv1_2")
        pool1   = tf.layers.max_pooling2d(conv1_2, pool_size=[1, 1], strides=2, name="pool1")

        # character-level lstm-cnn
        if self.num_rnn_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([self.create_rnn_cell() for _ in range(self.num_rnn_layers)], state_is_tuple=True)
        else:
            cell = self.create_rnn_cell()

        # initialize states
        initial_rnn_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # character-level lstm-cnn
        input_cnn = tf.reshape(conv1_2, [self.batch_size, 1, -1])

        rnn_input = [tf.squeeze(x, [1]) for x in tf.split(input_cnn, input_cnn.shape[1], 1)]
        outputs, final_rnn_state = tf.contrib.rnn.static_rnn(cell,
                                   rnn_input,
                                   initial_state=initial_rnn_state, dtype=tf.float32)

        # Get last time step's output feature for a "many to one" style classifier,
        # as in the image describing RNNs at the top of this page
        lstm_last_output = outputs[-1]

        # Linear activation
        logits = tf.matmul(lstm_last_output, self._weights['out']) + self._biases['out']

        prediction = tf.nn.softmax(logits)

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.output_, logits=logits))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)                    # Gradient Descent Optimizer
        train_op = optimizer.minimize(loss_op)

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.output_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return prediction, loss_op, accuracy, train_op
