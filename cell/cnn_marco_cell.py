import json
import os

from graphviz import Digraph

import cell


class CnnMarcoNode(cell.EnasNode):
  def __init__(self, index, operation=None, previous_indexes=None):
    self.index = index
    self.operation = operation
    self.previous_indexes = previous_indexes

  def __str__(self):
    instance_string = "Index: {}, operation: {}, previous_indexes: {}".format(
        self.index, self.operation, self.previous_indexes)
    return instance_string


class CnnMarcoModel(cell.EnasModel):
  def __init__(self):
    self.model_type = "cnn_marco"
    self.nodes = []

  @classmethod
  def load_from_json(cls, json_file_path):
    model_dict = json.load(open(json_file_path, "r"))

    cell_type = model_dict["cell_type"]
    nodes_dict = model_dict["nodes"]

    model = CnnMarcoModel()

    for node_dict in nodes_dict:
      index = node_dict.get("index", None)
      operation = node_dict.get("operation", None)
      previous_indexes = node_dict.get("previous_indexes", None)
      node = CnnMarcoNode(index, operation, previous_indexes)
      model.add_node(node)

    return model

  def draw_graph(self, file_path="graph"):
    dot = Digraph(format="png")

    total_node_number = len(self.nodes)
    # 1 for having next node and 0 for not
    have_next_node_array = [0 for i in range(total_node_number)]

    # Add nodes
    dot.node("Input", "Inputn")
    dot.node("Output0", "avg", color="green", style='filled')
    dot.node("Output1", "Softmax", color="lightblue", style='filled')

    # Add middle nodes
    for node in self.nodes:
      dot.node(str(node.index), "{}: {}".format(node.index, node.operation))

    # Add edges
    dot.edge("Input", "0")
    dot.edge("Output0", "Output1")

    # Add middle edges
    for node in self.nodes:
      if node.previous_indexes is not None:
        previous_index_list = [
            int(index) for index in node.previous_indexes.split(",")
        ]
        for previous_index in previous_index_list:
          dot.edge(str(previous_index), str(node.index))

          have_next_node_array[previous_index] = 1

    # Add edges for "no-next" nodes
    for i in range(total_node_number):
      if have_next_node_array[i] == 0:
        dot.edge(str(i), "Output0")

    # Save image file
    dot.render(file_path, view=True)
