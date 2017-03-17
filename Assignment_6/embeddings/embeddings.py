__author__ = 'pandaczm'

import math
import tensorflow as tf
import numpy as np
import random


class Skim_Gram_Embeddings:
    def __init__(self, batcher, vocabulary_size, embedding_size, num_sampled):
        self.batcher = batcher
        self.batch_size = batcher.batch_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape = [self.batch_size, 1])

        embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        softmax_weights = tf.Variable(tf.random_normal([self.vocabulary_size, self.embedding_size],stddev= 1.0 / math.sqrt(self.embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        embed = tf.nn.embedding_lookup(embeddings, self.train_dataset)
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights = softmax_weights, biases=softmax_biases,inputs=embed,num_sampled=self.num_sampled,    num_classes=self.vocabulary_size, labels=self.train_labels))
        self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keep_dims = True))
        self.normalized_embeddigns = embeddings / norm
        self.valid_examples = np.array(random.sample(range(100),16))
        valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddigns, valid_dataset)
        self.simlarity = tf.matmul(self.valid_embeddings, tf.transpose(self.normalized_embeddigns))

    # def __init__(self, batcher, embedding_size, vocabulary_size, num_sampled):
    #     self.batcher = batcher
    #     self.batch_size = batcher.batch_size
    #     self.vocabulary_size = vocabulary_size
    #     self.embedding_size = embedding_size
    #     self.num_sampled = num_sampled
    #
    #     self.graph = tf.Graph()
    #
    #     with self.graph.as_default(), tf.device('/cpu:0'):
    #         self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
    #         self.train_labels = tf.placeholder(tf.int32, shape = [self.batch_size, 1])
    #
    #         embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
    #         softmax_weights = tf.Variable(tf.random_normal([self.vocabulary_size, self.embedding_size],stddev= 1.0 / math.sqrt(self.embedding_size)))
    #         softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
    #
    #         embed = tf.nn.embedding_lookup(embeddings, self.train_dataset)
    #         self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights = softmax_weights, biases=softmax_biases,inputs=embed,num_sampled=self.num_sampled,    num_classes=self.vocabulary_size, labels=self.train_labels))
    #         self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)
    #
    #         norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keep_dims = True))
    #         self.normalized_embeddigns = embeddings / norm


    def session_run(self,session,train_data,train_label):
        return session.run([self.optimizer,self.loss], feed_dict={self.train_dataset:train_data, self.train_labels:train_label})








class CBOW_Embeddings:
    def __init__(self, batcher, vocabulary_size, embedding_size, num_sampled):
        self.batcher = batcher
        self.batch_size = batcher.batch_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape = [self.batch_size, 1])

        segment_ids = []
        for i in range(self.batch_size//self.batcher.num_skips):
            for j in range(2 * self.batcher.skip_window):
                segment_ids.append(i)

        self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape = [self.batch_size // self.batcher.num_skips, 1])

        embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        softmax_weights = tf.Variable(tf.random_normal([self.vocabulary_size, self.embedding_size],stddev= 1.0 / math.sqrt(self.embedding_size)))
        softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        embed = tf.nn.embedding_lookup(embeddings, self.train_dataset)
        CBOW_embed = tf.segment_sum(embed, segment_ids=tf.constant(segment_ids))
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights = softmax_weights, biases=softmax_biases,inputs=CBOW_embed,num_sampled=self.num_sampled,    num_classes=self.vocabulary_size, labels=self.train_labels))
        self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keep_dims = True))
        self.normalized_embeddigns = embeddings / norm
        self.valid_examples = np.array(random.sample(range(100),16))
        valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
        self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddigns, valid_dataset)
        self.simlarity = tf.matmul(self.valid_embeddings, tf.transpose(self.normalized_embeddigns))

    def session_run(self, session, train_data, train_label):
        return session.run([self.optimizer,self.loss], feed_dict = {self.train_dataset: train_data, self.train_labels:train_label})





def session_run(batcher, vocabulary_size, embedding_size, num_sampled, num_steps, batcher_type, id2gram):
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        if batcher_type == 'Skim Gram':
            m = Skim_Gram_Embeddings(batcher=batcher, vocabulary_size=vocabulary_size,embedding_size=embedding_size,num_sampled=num_sampled)
        else:
            m = CBOW_Embeddings(batcher=batcher, vocabulary_size=vocabulary_size,embedding_size=embedding_size,num_sampled=num_sampled)
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialzied')
        average_loss = 0
        for step in range(num_steps):
            if batcher_type == 'Skim Gram':
                batch_data, batch_labels = batcher.Skim_Gram_batch()
            else:
                batch_data, batch_labels = batcher.CBOW_batch()

            _, loss = m.session_run(session, batch_data,batch_labels)
            average_loss += loss
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                    print ('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0

            if step % 10000 == 0:
                sim = m.simlarity.eval()
                for i in range(16):
                    valid_word = id2gram[m.valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = id2gram[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)




        return m.normalized_embeddigns.eval()



## test ##
class Skim_Gram_Embeddings_test:
    # def __init__(self, batcher, vocabulary_size, embedding_size, num_sampled):
    #     self.batcher = batcher
    #     self.batch_size = batcher.batch_size
    #     self.vocabulary_size = vocabulary_size
    #     self.embedding_size = embedding_size
    #     self.num_sampled = num_sampled
    #     self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
    #     self.train_labels = tf.placeholder(tf.int32, shape = [self.batch_size, 1])
    #
    #     self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
    #     self.softmax_weights = tf.Variable(tf.random_normal([self.vocabulary_size, self.embedding_size],stddev= 1.0 / math.sqrt(self.embedding_size)))
    #     self.softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
    #
    #     embed = tf.nn.embedding_lookup(self.embeddings, self.train_dataset)
    #     self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights = self.softmax_weights, biases=self.softmax_biases,inputs=embed,num_sampled=self.num_sampled,    num_classes=self.vocabulary_size, labels=self.train_labels))
    #     self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)
    #
    #     norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings),1, keep_dims = True))
    #     self.normalized_embeddigns = self.embeddings / norm
    #     self.valid_examples = np.array(random.sample(range(100),16))
    #     valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
    #     self.valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddigns, valid_dataset)
    #     self.simlarity = tf.matmul(self.valid_embeddings, tf.transpose(self.normalized_embeddigns))
    def __init__(self, batcher, embedding_size, vocabulary_size, num_sampled):
        self.batcher = batcher
        self.batch_size = batcher.batch_size
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled

        self.graph = tf.Graph()

    def Session_graph(self,num_steps):

        with self.graph.as_default(), tf.device('/cpu:0'):
            train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
            train_labels = tf.placeholder(tf.int32, shape = [self.batch_size, 1])

            embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            softmax_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],stddev= 1.0 / math.sqrt(self.embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            embed = tf.nn.embedding_lookup(embeddings, train_dataset)
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights = softmax_weights, biases=softmax_biases,inputs=embed,num_sampled=self.num_sampled,    num_classes=self.vocabulary_size, labels=train_labels))
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1, keep_dims = True))
            normalized_embeddigns = embeddings / norm

        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            print ("initialized")
            average_loss = 0
            for step in range(num_steps):
                batch_data, batcher_labels = self.batcher.Skim_Gram_batch()
                feed_dict = {train_dataset:batch_data, train_labels:batcher_labels}
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                        print ('Average loss at step %d: %f' % (step, average_loss))
                        average_loss = 0
            return normalized_embeddigns.eval()
















