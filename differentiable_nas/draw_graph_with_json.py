#!/usr/bin/env python

import json
from graphviz import Digraph


def main():
  json_filename = "./dnas_arch.json"
  graph_filename = "./dnas_arch"
  draw_graph(json_filename, graph_filename)


def draw_graph(json_file_path, output_file_path):
  dot = Digraph(format="png")

  model_dict = json.load(open(json_file_path, "r"))

  cell_type = model_dict["cell_type"]
  nodes_dict = model_dict["nodes"]

  # Add input nodes
  dot.node("Input0", "x[t]")
  dot.node("Input1", "h[t-1]", color='lightblue', style='filled')
  dot.node("Output0", "avg", color="green", style='filled')
  dot.node("Output1", "h[t]", color="lightblue", style='filled')

  if cell_type == "dnn_one_previous":

    # Add nodes
    for node in nodes_dict:
      dot.node(
          str(node["index"]), "{}: {}".format(node["index"],
                                              node["operation"]))

    # Add edges
    total_node_number = len(nodes_dict)
    have_next_node_array = [0 for i in range(total_node_number)]
    for node in nodes_dict:
      if node["previous_index"] is not None:

        # Ignore the first previous index of first node
        if node["index"] == 0:
          pass
        else:
          dot.edge(str(node["previous_index"]), str(node["index"]))
          have_next_node_array[node["previous_index"]] = 1

  elif cell_type == "dnn_two_previous":
    # Add nodes
    for node in nodes_dict:
      dot.node(
          str(node["index"]), "[{}] {}:{}, {}:{}".format(
              node["index"], node["previous_index0"], node["operation0"],
              node["previous_index1"], node["operation1"]))

    # Add edges
    total_node_number = len(nodes_dict)
    have_next_node_array = [0 for i in range(total_node_number)]
    for node in nodes_dict:
      if node["previous_index0"] is not None and node["previous_index0"] != -1:
        dot.edge(str(node["previous_index0"]), str(node["index"]))
        have_next_node_array[node["previous_index0"]] = 1
      if node["previous_index1"] is not None and node["previous_index1"] != -1:
        dot.edge(str(node["previous_index1"]), str(node["index"]))
        have_next_node_array[node["previous_index1"]] = 1

  # Add input edges
  dot.edge("Input0", "0")
  dot.edge("Input1", "0")

  # Add edges for "no-next" nodes
  for i in range(total_node_number):
    if have_next_node_array[i] == 0:
      dot.edge(str(i), "Output0")

  # Add ouput edges
  dot.edge("Output0", "Output1")

  dot.render(output_file_path, view=True)


if __name__ == "__main__":
  main()
