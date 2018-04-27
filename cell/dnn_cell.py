import json
import os

from graphviz import Digraph

import cell


class DnnNode(cell.EnasNode):
  def __init__(self, index, previous_index=None, activation_function="tanh"):
    self.index = index
    self.previous_index = previous_index
    self.activation_function = activation_function

  def __str__(self):
    instance_string = "Index: {}, previous_index: {}, activation_function: {}".format(
        self.index, self.previous_index, self.activation_function)
    return instance_string


class DnnModel(cell.EnasModel):
  def __init__(self):
    self.model_type = "dnn"
    self.nodes = []

  @classmethod
  def load_from_json(cls, json_file_path):
    model_dict = json.load(open(json_file_path, "r"))

    cell_type = model_dict["cell_type"]
    nodes_dict = model_dict["nodes"]

    model = DnnModel()

    for node_dict in nodes_dict:
      index = node_dict.get("index", None)
      previous_index = node_dict.get("previous_index", None)
      activation_function = node_dict.get("activation_function", None)
      node = DnnNode(index, previous_index, activation_function)
      model.add_node(node)

    return model

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
