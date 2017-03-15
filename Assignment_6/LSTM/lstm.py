__author__ = 'pandaczm'

class LSTMGenerater(object):
    def __init__(self, lstmbatcher, num_nodes):
        self.lstmbatcher = lstmbatcher
        self.embedding_size =