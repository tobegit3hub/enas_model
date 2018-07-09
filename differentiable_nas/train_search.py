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

  epoch_number = 1
  input_size = 784
  output_size = 10

  dnn_hidden_unit = 16
  # Example: 0: tanh, 1: relu, 2: Sigmoid, 3: zero
  operation_number = 4

  node_number = 5
  # 0 + 1 + 2 + 3 + ... + n = (n * (n -1) / 2
  edge_number = int(node_number * (node_number - 1) / 2)
  reserve_node_edge_number = 2

  # The new hyperparameters as parameters for representation of activations
  hyperparameter_w = tf.get_variable(
      "hyperparameter_w", [edge_number, operation_number],
      dtype=tf.float32,
      initializer=tf.random_uniform_initializer)

  x = tf.placeholder(tf.float32, [None, 784])

  # The output of each node which aggregates all possible inputs and before activation
  node_output_array = [0.0 for i in range(node_number)]

  for j in range(node_number):
    if j == 0:
      weight = tf.get_variable(
          "w_input", [input_size, dnn_hidden_unit],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer)
      bias = tf.get_variable(
          "b_input", [dnn_hidden_unit],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer)
      layer = tf.matmul(x, weight) + bias
      node_output_array[j] = layer
    else:

      # The output of this node which aggregates all possible inputs
      node_output = 0.0

      for i in range(j):
        # node i -> node j, w01 means node 0 -> node 1
        weight = tf.get_variable(
            "w_{}_{}".format(i, j), [dnn_hidden_unit, dnn_hidden_unit],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer)
        bias = tf.get_variable(
            "b_{}_{}".format(i, j), [dnn_hidden_unit],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer)

        layer = node_output_array[i]
        # Get the hyperparameter_w item by i and j
        w_index = int(j * (j - 1) / 2) + i
        current_hyperparameter_w = hyperparameter_w[w_index]
        current_hyperparameter_w_softmax = tf.nn.softmax(
            current_hyperparameter_w)
        node_output_array[i] = current_hyperparameter_w_softmax[0] * tf.nn.tanh(
            layer) + current_hyperparameter_w_softmax[1] * tf.nn.tanh(
                layer) + current_hyperparameter_w_softmax[2] * tf.nn.sigmoid(
                    layer) + current_hyperparameter_w_softmax[3] * 0

        layer = tf.matmul(layer, weight) + bias
        node_output += layer

      node_output_array[j] = node_output

  # Aggreate all nodes's outputs as the model output
  model_output = 0.0
  for i in range(node_number):

    output_weight = tf.get_variable(
        "w_output_{}".format(i), [dnn_hidden_unit, output_size],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer)
    output_b = tf.get_variable(
        "b_output_{}".format(i), [output_size],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer)
    layer = node_output_array[i]
    layer = tf.matmul(layer, output_weight) + output_b
    model_output += layer

  y = model_output
  y_ = tf.placeholder(tf.int64, [None])
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

  with tf.Session() as sess:

    tf.global_variables_initializer().run()

    for i in range(938 * epoch_number):
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
    [[0.98046166 0.43295267 0.9859074  0.401366  ]
     [0.49687228 0.9334596  0.43843243 0.32343316]
     [0.0175321  0.45255908 0.177164   0.9900454 ]]
    """
    hyperparameter_w_value_list = hyperparameter_w_value.tolist()

    if reserve_node_edge_number == 1:
      output_architecture = {"cell_type": "dnn_one_previous", "nodes": []}
      for node_index in range(node_number):

        output_architecture_item = {
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

        output_architecture_item["index"] = node_index
        output_architecture_item["previous_index"] = top_k_max_edge_index
        output_architecture_item["operation"] = top_k_max_operator_index
        output_architecture["nodes"].append(output_architecture_item)

    elif reserve_node_edge_number == 2:
      output_architecture = {"cell_type": "dnn_two_previous", "nodes": []}
      for node_index in range(node_number):

        output_architecture_item = {
            "index": -1,
            "previous_index0": -1,
            "operation0": -1,
            "previous_index1": -1,
            "operation1": -1
        }

        if node_index == 0:
          current_hyperparameter = hyperparameter_w_value_list[0]
          current_max_value = max(current_hyperparameter)
          top_k_max_operator_index = current_hyperparameter.index(
              current_max_value)

          max_edge_index0 = -1
          max_edge_index1 = -1
          max_operator_index0 = top_k_max_operator_index
          max_operator_index1 = -1

        elif node_index == 1:
          current_hyperparameter0 = hyperparameter_w_value_list[0]
          current_max_value0 = max(current_hyperparameter0)
          top_k_max_operator_index0 = current_hyperparameter0.index(
              current_max_value0)

          current_hyperparameter1 = hyperparameter_w_value_list[1]
          current_max_value1 = max(current_hyperparameter1)
          top_k_max_operator_index1 = current_hyperparameter1.index(
              current_max_value1)

          max_edge_index0 = -1
          max_edge_index1 = 0
          max_operator_index0 = top_k_max_operator_index0
          max_operator_index1 = top_k_max_operator_index1

        else:

          # 0 + 1 + 2 + 3 + ... + n = n * (n-1) / 2
          start_index = int(node_index * (node_index - 1) / 2)
          end_index = int(start_index + node_index)

          max_value0 = -1
          max_value1 = -1
          max_edge_index0 = -1
          max_edge_index1 = -1
          max_operator_index0 = -1
          max_operator_index1 = -1

          for i, current_hyperparameter in enumerate(
              hyperparameter_w_value_list[start_index:end_index]):

            # Example: [0.3108067810535431, 0.8650654554367065, 0.14078158140182495, 0.371991902589798]
            current_max_value = max(current_hyperparameter)
            current_max_operator_index = current_hyperparameter.index(
                current_max_value)

            if current_max_value > max_value0:
              max_value0 = current_max_value
              max_edge_index0 = i
              max_operator_index0 = current_max_operator_index
            elif current_max_value > max_value1:
              max_value1 = current_max_value
              max_edge_index1 = i
              max_operator_index1 = current_max_operator_index

        output_architecture_item["index"] = node_index
        output_architecture_item["previous_index0"] = max_edge_index0
        output_architecture_item["operation0"] = max_operator_index0
        output_architecture_item["previous_index1"] = max_edge_index1
        output_architecture_item["operation1"] = max_operator_index1
        output_architecture["nodes"].append(output_architecture_item)

    else:
      logging.error("Unsupported reserve_node_edge_number: {}".format(
          reserve_node_edge_number))

    logging.info("Final architecture: {}".format(output_architecture))
    architecture_json_filename = "./dnas_arch.json"
    with open(architecture_json_filename, "w") as f:
      json.dump(output_architecture, f)


if __name__ == "__main__":
  main()
