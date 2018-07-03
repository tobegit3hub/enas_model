#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def main():

  mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data")

  NODE_NUMBER = 3
  LIMIT_EDGE = 1

  hyperparameter_w = tf.get_variable(
      "hyperparameter_w", [3, 4],
      dtype=tf.float32,
      initializer=tf.random_uniform_initializer)

  x = tf.placeholder(tf.float32, [None, 784])

  w0 = tf.get_variable(
      "w0", [784, 64],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer)
  b0 = tf.get_variable(
      "b0", [64],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer)
  layer = tf.matmul(x, w0) + b0
  #layer = tf.nn.relu(layer)
  current_hyperparameter_w = hyperparameter_w[0]
  current_hyperparameter_w_softmax = tf.nn.softmax(current_hyperparameter_w)
  layer = current_hyperparameter_w_softmax[0] * tf.nn.tanh(
      layer) + current_hyperparameter_w_softmax[1] * tf.nn.tanh(
          layer) + current_hyperparameter_w_softmax[2] * tf.nn.sigmoid(
              layer) + current_hyperparameter_w_softmax[3] * 0

  w1 = tf.get_variable(
      "w1", [64, 32],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer)
  b1 = tf.get_variable(
      "b1", [32],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer)
  layer = tf.matmul(layer, w1) + b1
  #layer = tf.nn.relu(layer)
  current_hyperparameter_w = hyperparameter_w[1]
  current_hyperparameter_w_softmax = tf.nn.softmax(current_hyperparameter_w)
  layer = current_hyperparameter_w_softmax[0] * tf.nn.tanh(
      layer) + current_hyperparameter_w_softmax[1] * tf.nn.tanh(
          layer) + current_hyperparameter_w_softmax[2] * tf.nn.sigmoid(
              layer) + current_hyperparameter_w_softmax[3] * 0

  w2 = tf.get_variable(
      "w2", [32, 16],
      dtype=tf.float32,
      initializer=tf.random_uniform_initializer)
  b2 = tf.get_variable(
      "b2", [16], dtype=tf.float32, initializer=tf.random_uniform_initializer)
  layer = tf.matmul(layer, w2) + b2
  #layer = tf.nn.relu(layer)
  current_hyperparameter_w = hyperparameter_w[2]
  current_hyperparameter_w_softmax = tf.nn.softmax(current_hyperparameter_w)
  layer = current_hyperparameter_w_softmax[0] * tf.nn.tanh(
      layer) + current_hyperparameter_w_softmax[1] * tf.nn.tanh(
          layer) + current_hyperparameter_w_softmax[2] * tf.nn.sigmoid(
              layer) + current_hyperparameter_w_softmax[3] * 0

  w3 = tf.get_variable(
      "w3", [16, 10],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer)
  b3 = tf.get_variable(
      "b3", [10],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer)
  y = tf.matmul(layer, w3) + b3

  y_ = tf.placeholder(tf.int64, [None])
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  with tf.Session() as sess:

    tf.global_variables_initializer().run()

    #writer = tf.summary.FileWriter(FLAGS.output_path, sess.graph)

    for i in range(938):
      #batch_xs, batch_ys = mnist.train.next_batch(100)
      batch_xs, batch_ys = mnist.train.next_batch(64)
      _, loss_value = sess.run(
          [train_step, cross_entropy], feed_dict={x: batch_xs,
                                                  y_: batch_ys})

      #if i % 50 == 0:
      #    print("Run batch")

      if i % 1000 == 0:

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_value = sess.run(
            accuracy, feed_dict={x: mnist.test.images,
                                 y_: mnist.test.labels})
        print("Loss: {}".format(loss_value))
        print("Accuracy: {}".format(accuracy_value))

        hyperparameter_w_value = sess.run(hyperparameter_w)
        print("hyperparameter_w_value: {}".format(hyperparameter_w_value))

    #import ipdb;ipdb.set_trace()
    """
    Get the top-k edges indexes and operator indexes
    [[0.98046166 0.43295267 0.9859074  0.401366  ]
     [0.49687228 0.9334596  0.43843243 0.32343316]
     [0.0175321  0.45255908 0.177164   0.9900454 ]]
    """

    hyperparameter_w_value_list = hyperparameter_w_value.tolist()
    output_architecture_list = []
    for node_index in range(NODE_NUMBER):

      output_architecture = {"previous_node": -1, "operation": -1}

      if node_index == 0:
        pass
      else:
        import ipdb
        ipdb.set_trace()
        # 0 + 1 + 2 + 3 + ... + n = n * (n-1) / 2
        start_index = int(node_index * (node_index - 1) / 2)
        end_index = int(start_index + node_index)

        top_k_max_value = -1
        top_k_max_edge_index = -1
        top_k_max_operator_index = -1

        for i, current_hyperparameter in enumerate(
            hyperparameter_w_value_list[start_index:end_index]):

          # Example: [0.3108067810535431, 0.8650654554367065, 0.14078158140182495, 0.371991902589798]
          current_max_value = max(current_hyperparameter)
          current_max_operator_index = current_hyperparameter.index(
              current_max_value)

          if current_max_value > top_k_max_value:
            top_k_max_value = current_max_value
            top_k_max_edge_index = i
            top_k_max_operator_index = current_max_operator_index

        output_architecture["previous_node"] = top_k_max_edge_index
        output_architecture["operation"] = top_k_max_operator_index
        output_architecture_list.append(output_architecture)
        import ipdb
        ipdb.set_trace()


if __name__ == "__main__":
  main()
