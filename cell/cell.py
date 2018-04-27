class EnasNode(object):
  def __init__(self):
    self.index = None

  def __str__(self):
    instance_string = "Index: {}".format(self.index)
    return instance_string

  def __repr__(self):
    return self.__str__()


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
