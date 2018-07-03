#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import json
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def main():

  mnist = input_data.read_data_sets("/tmp/tensorflow/mnist/input_data")

  input_size = 784
  output_size = 10

  dnn_hidden_unit = 16
  # Example: 0: tanh, 1: relu, 2: Sigmoid, 3: zero
  operation_number = 4

  node_number = 5
  # 0 + 1 + 2 + 3 + ... + n = (n**2 + n) / 2
  edge_number = int((node_number**2 + node_number) / 2)
  reserve_node_edge_number = 1

  hyperparameter_w = tf.get_variable(
      "hyperparameter_w", [edge_number, operation_number],
      dtype=tf.float32,
      initializer=tf.random_uniform_initializer)

  x = tf.placeholder(tf.float32, [None, 784])

  for i in range(node_number):
    if i == 0:
      weight = tf.get_variable(
          "w_{}".format(i), [input_size, dnn_hidden_unit],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer)
      bias = tf.get_variable(
          "b_{}".format(i), [dnn_hidden_unit],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer)
      layer = tf.matmul(x, weight) + bias
    else:
      weight = tf.get_variable(
          "w_{}".format(i), [dnn_hidden_unit, dnn_hidden_unit],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer)
      bias = tf.get_variable(
          "b_{}".format(i), [dnn_hidden_unit],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer)
      layer = tf.matmul(layer, weight) + bias

    #layer = tf.nn.relu(layer)
    current_hyperparameter_w = hyperparameter_w[0]
    current_hyperparameter_w_softmax = tf.nn.softmax(current_hyperparameter_w)
    layer = current_hyperparameter_w_softmax[0] * tf.nn.tanh(
        layer) + current_hyperparameter_w_softmax[1] * tf.nn.tanh(
            layer) + current_hyperparameter_w_softmax[2] * tf.nn.sigmoid(
                layer) + current_hyperparameter_w_softmax[3] * 0

  output_weight = tf.get_variable(
      "w_output", [dnn_hidden_unit, output_size],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer)
  output_b = tf.get_variable(
      "b_output", [output_size],
      dtype=tf.float32,
      initializer=tf.truncated_normal_initializer)
  y = tf.matmul(layer, output_weight) + output_b

  y_ = tf.placeholder(tf.int64, [None])
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  with tf.Session() as sess:

    tf.global_variables_initializer().run()

    #writer = tf.summary.FileWriter(FLAGS.output_path, sess.graph)

    for i in range(9380):
      #batch_xs, batch_ys = mnist.train.next_batch(100)
      batch_xs, batch_ys = mnist.train.next_batch(64)
      _, loss_value = sess.run(
          [train_step, cross_entropy], feed_dict={x: batch_xs,
                                                  y_: batch_ys})

      if i % 1000 == 0:

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_value = sess.run(
            accuracy, feed_dict={x: mnist.test.images,
                                 y_: mnist.test.labels})
        logging.info("Loss: {}".format(loss_value))
        logging.info("Accuracy: {}".format(accuracy_value))

        hyperparameter_w_value = sess.run(hyperparameter_w)
        logging.info(
            "hyperparameter_w_value: {}".format(hyperparameter_w_value))
    """
    Get the top-k edges indexes and operator indexes
    
    [[0.98046166 0.43295267 0.9859074  0.401366  ]
     [0.49687228 0.9334596  0.43843243 0.32343316]
     [0.0175321  0.45255908 0.177164   0.9900454 ]]
    """

    hyperparameter_w_value_list = hyperparameter_w_value.tolist()
    output_architecture_list = {"cell_type": "dnn", "nodes": []}
    for node_index in range(node_number):

      output_architecture = {
          "index": -1,
          "previous_index": -1,
          "operation": -1
      }

      if node_index == 0:
        current_hyperparameter = hyperparameter_w_value_list[0]

        top_k_max_edge_index = -1
        current_max_value = max(current_hyperparameter)
        top_k_max_operator_index = current_hyperparameter.index(
            current_max_value)

      else:

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

      output_architecture["index"] = node_index
      output_architecture["previous_index"] = top_k_max_edge_index
      output_architecture["operation"] = top_k_max_operator_index
      output_architecture_list["nodes"].append(output_architecture)

    logging.info("Final architecture: {}".format(output_architecture_list))
    architecture_json_filename = "./dnas_arch.json"
    with open(architecture_json_filename, "w") as f:
      json.dump(output_architecture_list, f)


if __name__ == "__main__":
  main()
