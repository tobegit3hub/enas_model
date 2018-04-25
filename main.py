#!/usr/bin/env python

import os
import json
import logging

import numpy as np
import tensorflow as tf

import cells


def main():
  trainRnnModel = TrainRnnModel()
  trainRnnModel.train()


class TrainRnnModel(object):
  def __init__(self):
    self.name_variabel_map = {}

  def train(self):
    json_file_path = "./examples/rnn_example.json"
    rnn_model = cells.RnnModel.load_from_json(json_file_path)

    print("Build model op")

    #x_train = np.ones((32, 784))
    x_train = np.random.rand(32, 784)
    #y_train = np.zeros((32, 10))
    y_train = np.random.randint(0, 2, size=(32, 10))

    input_feature_size = 784
    output_label_size = 10
    learning_rate = 0.5
    epoch_number = 10
    batch_size = 1
    buffer_size = 10
    step_to_validate = 1
    tensorboard_path = "./tensorboard"
    if os.path.exists(tensorboard_path) == False:
      os.makedirs(tensorboard_path)
    checkpoint_path = "./checkpoint"
    if os.path.exists(checkpoint_path) == False:
      os.makedirs(checkpoint_path)
    checkpoint_file = checkpoint_path + "/checkpoint.ckpt"
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)

    x_placeholder = tf.placeholder(tf.float32, [None, input_feature_size])
    y_placeholder = tf.placeholder(tf.float32, [None, output_label_size])

    print("Build train graph")

    global_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name="global_step")

    # input layer
    """
    output0 = fc_layer(x_placeholder, input_feature_size, input_feature_size, activation="tanh", index=0)
    output1 = fc_layer(output0, input_feature_size, input_feature_size, activation="tanh", index=1)
    output2 = fc_layer(output1, input_feature_size, input_feature_size, activation="tanh", index=2)
    output7 = fc_layer(output1, input_feature_size, input_feature_size, activation="tanh", index=7)
    average_output = (output2 + output7) / 2
    """

    total_node_number = len(rnn_model.nodes)

    # Example: {"0": Output0, "1": Output1, "2: Output2}
    index_output_map = {}

    for i in range(total_node_number):
      # TODO: Need to make sure that nodes are stores with index in array
      node = rnn_model.nodes[i]

      if i == 0:
        input = x_placeholder
      else:
        input = index_output_map.get(str(node.previous_index), None)

      output = self.fc_layer(
          input,
          input_feature_size,
          input_feature_size,
          activation_function=node.activation_function,
          index=node.index)
      index_output_map[str(i)] = output

    # Example: [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0]
    have_next_node_array = [0 for i in range(total_node_number)]
    for node in rnn_model.nodes:
      if node.previous_index is not None:
        have_next_node_array[node.previous_index] = 1

    average_output = None
    not_have_next_total_number = 0
    for i in range(total_node_number):
      if have_next_node_array[i] == 0:
        not_have_next_total_number += 1
        if average_output == None:
          average_output = index_output_map.get(str(i), None)
        else:
          average_output += index_output_map.get(str(i), None)

    average_output = average_output / not_have_next_total_number

    # output layer
    input = average_output
    output_w = tf.get_variable(
        "output_w", [input_feature_size, output_label_size],
        dtype=tf.float32,
        initializer=tf.zeros_initializer)
    output_b = tf.get_variable(
        "output_b", [output_label_size],
        dtype=tf.float32,
        initializer=tf.zeros_initializer)
    logits = tf.matmul(input, output_w) + output_b

    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_placeholder))
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y_placeholder))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)

    # Save Variables with names in order to restore from other architectures
    saver = tf.train.Saver(self.name_variabel_map)
    tf.summary.scalar("global_step", global_step)
    summary_op = tf.summary.merge_all()

    # Start training
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

      self.restore_from_checkpoint(sess, saver, latest_checkpoint)

      try:

        for i in range(epoch_number):

          _, loss_value, step_value = sess.run(
              [train_op, loss, global_step],
              feed_dict={x_placeholder: x_train,
                         y_placeholder: y_train})

          if step_value % step_to_validate == 0:
            summary_value = sess.run(summary_op)
            print("Run step: {}, loss: {}".format(step_value, loss_value))

            writer.add_summary(summary_value, step_value)
            saver.save(sess, checkpoint_file, global_step=step_value)

      except tf.errors.OutOfRangeError:
        print("End of training")

  def restore_from_checkpoint(self, sess, saver, checkpoint):
    if checkpoint:
      logging.info("Restore session from checkpoint: {}".format(checkpoint))
      saver.restore(sess, checkpoint)
      return True
    else:
      logging.warn("Checkpoint not found: {}".format(checkpoint))
      return False

  def fc_layer(self,
               input,
               input_shape,
               output_shape,
               activation_function="tanh",
               index=None):
    """
    weight = tf.get_variable(
        "weight_{}".format(index), [input_shape, output_shape],
        dtype=tf.float32,
        initializer=tf.zeros_initializer)
    bias = tf.get_variable(
        "bias_{}".format(index), [output_shape],
        dtype=tf.float32,
        initializer=tf.zeros_initializer)
    """
    weight = tf.get_variable(
        "weight_{}".format(index), [input_shape, output_shape],
        dtype=tf.float32,
        initializer=None)
    bias = tf.get_variable(
        "bias_{}".format(index), [output_shape],
        dtype=tf.float32,
        initializer=None)

    self.name_variabel_map["weight_{}".format(index)] = weight
    self.name_variabel_map["bias_{}".format(index)] = bias

    output = tf.matmul(input, weight) + bias
    if activation_function == "tanh":
      output = tf.nn.tanh(output)
    elif activation_function == "relu":
      output = tf.nn.relu(output)
    elif activation_function == "identity":
      output = output
    elif activation_function == "sigmoid":
      output = tf.nn.sigmoid(output)
    else:
      output = tf.nn.relu(output)
    return output


if __name__ == "__main__":
  main()
