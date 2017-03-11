__author__ = 'pandaczm'

import math
import tensorflow as tf


class Embeddings:
    def __init__(self, batcher, vocabulary_size, embedding_size, num_sampled):
        self.batcher = batcher
        self.batch_size = batcher.batch_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.graph = tf.Graph()

    def Skip_Gram_Graph(self):
        with self.graph.as_default(), tf.device('/cpu:0'):
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape = [self.batch_size, 1])


