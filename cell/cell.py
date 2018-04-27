#!/usr/bin/env python

import json
import os

from graphviz import Digraph


class EnasNode(object):
  def __init__(self):
    self.index = None


class EnasModel(object):
  def __init__(self):
    self.model_type = None
    self.nodes = []

  def add_node(self, node):
    self.nodes.append(node)

  def __str__(self):
    instance_string = "Type: {}, nodes: {}".format(self.model_type, self.nodes)
    return instance_string

  def __repr__(self):
    return self.__str__()


class RnnNode(EnasNode):
  def __init__(self, index, previous_index=None, activation_function="tanh"):
    self.index = index
    self.previous_index = previous_index
    self.activation_function = activation_function

  def __str__(self):
    instance_string = "index: {}, previous_index: {}, activation_function: {}".format(
        self.index, self.previous_index, self.activation_function)
    return instance_string

  def __repr__(self):
    return self.__str__()


class RnnModel(EnasModel):
  def __init__(self):
    self.model_type = "rnn"
    self.nodes = []

  @classmethod
  def load_from_json(cls, json_file_path):
    model_dict = json.load(open(json_file_path, "r"))

    cell_type = model_dict["cell_type"]
    nodes_dict = model_dict["nodes"]

    rnn_model = RnnModel()

    for node_dict in nodes_dict:
      index = node_dict.get("index", None)
      previous_index = node_dict.get("previous_index", None)
      activation_function = node_dict.get("activation_function", None)
      rnn_node = RnnNode(index, previous_index, activation_function)
      rnn_model.add_node(rnn_node)

    return rnn_model

  def __str__(self):
    instance_string = "nodes: {}".format(self.nodes)
    return instance_string

  def __repr__(self):
    return self.__str__()

  def draw_graph(self, file_path="graph"):
    dot = Digraph(format="png")

    # Add nodes
    dot.node("Input0", "x[t]")
    dot.node("Input1", "h[t-1]", color='lightblue', style='filled')
    dot.node("Output0", "avg", color="green", style='filled')
    dot.node("Output1", "h[t]", color="lightblue", style='filled')

    for node in self.nodes:
      dot.node(
          str(node.index), "{}: {}".format(node.index,
                                           node.activation_function))

    # Add edges
    total_node_number = len(self.nodes)
    have_next_node_array = [0 for i in range(total_node_number)]
    for node in self.nodes:
      if node.previous_index is not None:
        dot.edge(str(node.previous_index), str(node.index))

        have_next_node_array[node.previous_index] = 1

    # Add edge for input node
    dot.edge("Input0", "0")
    dot.edge("Input1", "0")

    # Add edges for "no-next" nodes
    for i in range(total_node_number):
      if have_next_node_array[i] == 0:
        dot.edge(str(i), "Output0")

    dot.edge("Output0", "Output1")

    dot.render(file_path, view=True)


def main():
  print("Start")

  json_file_path = "./examples/rnn_example.json"
  rnn_model = RnnModel.load_from_json(json_file_path)
  rnn_model.draw_graph("rnn_example")


if __name__ == "__main__":
  main()