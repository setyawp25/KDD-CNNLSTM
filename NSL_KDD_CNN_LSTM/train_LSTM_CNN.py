import os
import h5py
import time
import numpy as np
import core.utils as utils
from core.model3 import LSTM
import tensorflow as tf

# gpu_device = ['0', '1']
gpu_device = ['0']
print 'gpu_device = %s' % (','.join(gpu_device))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_device)

flags = tf.flags

# model params
flags.DEFINE_integer('hidden_size',         640,    'size of LSTM internal state')
flags.DEFINE_integer('rnn_layers',          1,      'number of layers in the LSTM')
flags.DEFINE_integer('num_unroll_steps',    1,      'number of timesteps to unroll for = number of words to process per process')
flags.DEFINE_integer('batch_size',          32,     'number of sequences to train on in parallel')
flags.DEFINE_float('learning_rate',       0.01,  'learning rate')
flags.DEFINE_integer('num_class',            5,     'number of classes')
# flags.DEFINE_integer('training_loop_times', 300,    '')
# flags.DEFINE_integer('training_steps',    22544,    'training steps')
flags.DEFINE_integer('max_epochs',           100,    'number of full passes through the training data')
flags.DEFINE_float  ('dropout',             0.5,    'dropout. 0 = no dropout')
flags.DEFINE_float  ('lambda_loss_amount',  0.0015, 'lambda loss amount')

flags.DEFINE_integer  ('input_size',            64,     'column number of input data')
flags.DEFINE_integer  ('img_rows',            2,     'row number of input data')  # 32*31 = 992 = RSSI feature columns

# bookkeeping
# flags.DEFINE_integer('display_step',        200,    'display step')
flags.DEFINE_integer('print_every',    1000,    'how often to print current loss')

FLAGS = flags.FLAGS

def load_train_test_data(path, h5py_fname):

    hf = h5py.File(os.path.join(path, h5py_fname), 'r')
    print "h5py keys = ", hf.keys()

    X_train = np.array(hf.get('dataset_X_train'))
    X_test = np.array(hf.get('dataset_X_test'))
    Y_train = np.array(hf.get('dataset_Y_train'))
    Y_test = np.array(hf.get('dataset_Y_test'))

    return X_train, X_test, Y_train, Y_test

def main():
    dataset_dir_processed = "NSL-KDD_dataset_processed"
    h5py_fname = 'NSL-KDD_dataset_processed.h5'
    X_train, X_test, Y_train, Y_test = load_train_test_data(dataset_dir_processed, h5py_fname)
    training_data_count, timesteps_size, input_size = X_train.shape   # input_size = feature dimensions = the dimension of each 'row' = X_train.shape[1]
    print "Training_data_count = %s,\tTimesteps_size = %s,\tInput_size = %s" % (training_data_count, timesteps_size, input_size)
    print "X_train.shape = %s\tY_train.shape = %s" % (X_train.shape, Y_train.shape)
    print "X_test.shape = %s\tY_test.shape = %s" % (X_test.shape, Y_test.shape)

    batch_size = tf.placeholder(tf.int32, [])
    x_train_reader = utils.DataReader(X_train, Y_train, FLAGS.batch_size, input_size, FLAGS.num_unroll_steps)
    x_test_reader  = utils.DataReader(X_test, Y_test, FLAGS.batch_size, input_size, FLAGS.num_unroll_steps)

    lstm_model = LSTM(_learning_rate=FLAGS.learning_rate,
                    _batch_size=FLAGS.batch_size,
                    dropout=FLAGS.dropout,
                    hidden_size=FLAGS.hidden_size,
                    num_input=input_size,
                    num_rnn_layers=FLAGS.rnn_layers,
                    num_unroll_steps=FLAGS.num_unroll_steps)

    prediction, loss_op, accuracy, train_op = lstm_model.LSTM_RNN()

    # Start training
    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    # Launch the graph

    # Perform Training steps with "batch_size" amount of example data at each loop
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    # with tf.Graph().as_default(), tf.Session() as sess:
        init = tf.global_variables_initializer()
        # Run the initializer
        sess.run(init)
        # print sess.run(init)
        for epoch in range(FLAGS.max_epochs):
            epoch_start_time = time.time()
            count = 0
            x_train_reader.shuffle_arr_list()
            for x_batch, y_batch in x_train_reader.iter_batches():
                # print 'x_batch.shape = %s \t y_batch.shape = %s'% (x_batch.shape, y_batch.shape)
                count += 1
                start_time = time.time()

                #_, loss, acc  = sess.run([train_op, loss_op, accuracy], feed_dict={lstm_model.input_: x_batch, lstm_model.output_: y_batch, lstm_model.batch_size: FLAGS.batch_size})
                _, loss, acc  = sess.run([train_op, loss_op, accuracy], feed_dict={lstm_model.input_: x_batch, lstm_model.output_: y_batch, batch_size: FLAGS.batch_size})

                time_elapsed = time.time() - start_time
                if count % FLAGS.print_every == 0:
                    print('%3d: [%5d/%5d], train_loss = %.8f,\taccuracy = %.7f,\tsecs/batch = %.4fs' % (epoch,
                                                            count,
                                                            x_train_reader.length,
                                                            loss, acc,
                                                            time_elapsed))
            # Calculate accuracy for testing data
            print '%s Epoch training time: %s %s' % ('='*10, time.time()-epoch_start_time, '='*10)

            for x_test, y_test in x_test_reader.iter_batches():

                _, loss, acc  = sess.run([train_op, loss_op, accuracy], feed_dict={lstm_model.input_: x_test, lstm_model.output_: y_test, batch_size: FLAGS.batch_size})
                test_accuracies.append(acc)
                test_losses.append(loss)

            average_acc = sum(test_accuracies)/len(test_accuracies)
            average_loss = sum(test_losses)/len(test_losses)

            print('\t >>> test_loss = %.8f,\taccuracy = %.7f' % (average_loss, average_acc))


        print("Optimization Finished!")

        # Accuracy for test data
        print("\n> Final Testing Accuracy: %s" % (average_acc))

        # Calculate accuracy for testing data
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={x: X_test, y: Y_test, batch_size: Y_test.shape[0]}))


if __name__ == "__main__":
    main()
