#!/usr/bin/env python

from cell import dnn_cell
from cell import rnn_cell
from cell import cnn_marco_cell
from cell import cnn_micro_cell


def main():
  #draw_dnn_cell()
  #draw_rnn_cell()
  draw_cnn_marco_cell()


def draw_dnn_cell():
  json_file_path = "./examples/dnn_example.json"
  model = dnn_cell.DnnModel.load_from_json(json_file_path)
  model.draw_graph("./examples/dnn_example")


def draw_rnn_cell():
  json_file_path = "./examples/rnn_example.json"
  model = rnn_cell.RnnModel.load_from_json(json_file_path)
  model.draw_graph("./examples/rnn_example")


def draw_cnn_marco_cell():
  json_file_path = "./examples/cnn_marco_example.json"
  model = cnn_marco_cell.CnnMarcoModel.load_from_json(json_file_path)
  model.draw_graph("./examples/cnn_marco_example")


if __name__ == "__main__":
  main()
