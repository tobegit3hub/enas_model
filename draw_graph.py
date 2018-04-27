#!/usr/bin/env python

from cell import dnn_cell


def main():
  json_file_path = "./examples/dnn_example.json"
  rnn_model = dnn_cell.DnnModel.load_from_json(json_file_path)
  rnn_model.draw_graph("./examples/dnn_example")


if __name__ == "__main__":
  main()
