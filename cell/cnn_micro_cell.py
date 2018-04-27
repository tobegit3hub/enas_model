import json
import os

from graphviz import Digraph

import cell


class CnnMicroNode(cell.EnasNode):
  def __init__(self,
               index,
               previous_index1=None,
               operation1=None,
               previous_index2=None,
               operation2=None):
    self.index = index
    self.previous_index1 = previous_index1
    self.operation1 = operation1
    self.previous_index2 = previous_index2
    self.operation2 = operation2

  def __str__(self):
    instance_string = "Index: {}, previous_index1: {}, operation1: {}, previous_index2: {}, operation2: {}".format(
        self.index, self.previous_index1, self.operation1,
        self.previous_index2, self.operation2)
    return instance_string


class CnnMicroModel(cell.EnasModel):
  def __init__(self):
    self.model_type = "cnn_micro"
    self.nodes = []

  @classmethod
  def load_from_json(cls, json_file_path):
    model_dict = json.load(open(json_file_path, "r"))

    cell_type = model_dict["cell_type"]
    nodes_dict = model_dict["nodes"]

    model = CnnMicroModel()

    for node_dict in nodes_dict:
      index = node_dict.get("index", None)
      previous_index1 = node_dict.get("previous_index1", None)
      operation1 = node_dict.get("operation1", None)
      previous_index2 = node_dict.get("previous_index2", None)
      operation2 = node_dict.get("operation2", None)
      node = CnnMicroNode(index, previous_index1, operation1, previous_index2,
                          operation2)
      model.add_node(node)

    return model

  def draw_graph(self, file_path="graph"):
    dot = Digraph(format="png")

    # Add the node 0 and 1 which is not described in the json
    total_node_number = len(self.nodes) + 2
    have_next_node_array = [0 for i in range(total_node_number)]

    # Add nodes
    dot.node("0", "0")
    dot.node("1", "1")
    dot.node("Output0", "avg", color="green", style='filled')
    dot.node("Output1", "Output", color="lightblue", style='filled')

    for node in self.nodes:
      dot.node(
          str(node.index), "{}: {}({}), {}({})".format(
              node.index, node.operation1, node.previous_index1,
              node.operation2, node.previous_index2))

    # Add edges
    for node in self.nodes:
      if node.previous_index1 is not None:
        dot.edge(str(node.previous_index1), str(node.index))

        have_next_node_array[node.previous_index1] = 1
      if node.previous_index2 is not None:
        dot.edge(str(node.previous_index2), str(node.index))

        have_next_node_array[node.previous_index2] = 1

    # Add edges for "no-next" nodes
    for i in range(total_node_number):
      if have_next_node_array[i] == 0:
        dot.edge(str(i), "Output0")

    dot.edge("Output0", "Output1")

    dot.render(file_path, view=True)
