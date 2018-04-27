import dnn_cell


class RnnNode(dnn_cell.DnnNode):
  pass


class RnnModel(dnn_cell.DnnModel):
  # TODO: Implementation should be different form dnn cell
  def __init__(self):
    self.model_type = "rnn"
    self.nodes = []
