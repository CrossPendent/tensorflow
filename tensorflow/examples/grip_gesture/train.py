from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import grip_input_data

import tensorflow as tf

RAW_DATA_SIZE = 450
g_labels = ['other', 'slide_up', 'slide_down']

def main(_):
  # Import data
  grip_data = grip_input_data.GripData('./', g_labels)

  # Create the model
  x = tf.placeholder(tf.float32, [None, RAW_DATA_SIZE])
  W = tf.Variable(tf.zeros([RAW_DATA_SIZE, len(g_labels)]))
  b = tf.Variable(tf.zeros([len(g_labels)]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.int64, [None])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.losses.sparse_softmax_cross_entropy on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = grip_data.getNextTrainBatch(100) # mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), y_)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  test_data = grip_data.getTestData()
  print(sess.run(accuracy, feed_dict={x: test_data['x'],
                                      y_: test_data['y']}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
