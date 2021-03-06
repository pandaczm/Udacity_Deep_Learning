__author__ = 'pandaczm'

import tensorflow as tf
import numpy as np
import random
import string



class LSTMGenerater(object):
    def __init__(self, embeddings_size,vocabulary_size, batch_size, num_unrollings, num_nodes, num_sampled, drop_out = False, keep_prob = 0.5):

        self.embedding_size = embeddings_size
        self.vocabulary_size = vocabulary_size
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings
        self.num_sampled = num_sampled


        ix = tf.Variable(tf.truncated_normal([self.embedding_size, self.num_nodes], -0.1, 0.1))
        im = tf.Variable(tf.truncated_normal([self.num_nodes, self.num_nodes], -0.1, 0.1))
        ib = tf.Variable(tf.zeros([1, self.num_nodes]))
        fx = tf.Variable(tf.truncated_normal([self.embedding_size, self.num_nodes], -0.1, 0.1))
        fm = tf.Variable(tf.truncated_normal([self.num_nodes, self.num_nodes], -0.1, 0.1))
        fb = tf.Variable(tf.zeros([1, self.num_nodes]))
        # Memory cell: input, state and bias.
        cx = tf.Variable(tf.truncated_normal([self.embedding_size, self.num_nodes], -0.1, 0.1))
        cm = tf.Variable(tf.truncated_normal([self.num_nodes, self.num_nodes], -0.1, 0.1))
        cb = tf.Variable(tf.zeros([1, num_nodes]))
        # Output gate: input, previous output, and bias.
        ox = tf.Variable(tf.truncated_normal([self.embedding_size, self.num_nodes], -0.1, 0.1))
        om = tf.Variable(tf.truncated_normal([self.num_nodes, self.num_nodes], -0.1, 0.1))
        ob = tf.Variable(tf.zeros([num_nodes]))

        saved_output = tf.Variable(tf.zeros([self.batch_size, self.num_nodes]), trainable=False)
        saved_state = tf.Variable(tf.zeros([self.batch_size, self.num_nodes]), trainable=False)

        w = tf.Variable(tf.truncated_normal([self.num_nodes, self.vocabulary_size], -0.1, 0.1))
        b = tf.Variable(tf.zeros([self.vocabulary_size]))



        self.train_data = list()
        for _ in range(self.num_unrollings + 1):
            self.train_data.append(tf.placeholder(tf.float32, shape = [self.batch_size, self.embedding_size]))
        train_inputs = self.train_data[: self.num_unrollings]
        # train_labels = self.train_data[1:]
        self.train_label = list()
        for _ in range(self.num_unrollings + 1):
            self.train_label.append(tf.placeholder(tf.float32, shape = [self.batch_size, 1]))
        train_labels = self.train_label[1:]

        def lstm_cell(i, o, state):

            input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
            forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
            update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
            state = forget_gate * state + input_gate * tf.tanh(update)
            output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
            return output_gate * tf.tanh(state), state

        outputs = list()
        output = saved_output
        state = saved_state
        for i in train_inputs:
            if drop_out:
                i = tf.nn.dropout(i, keep_prob=keep_prob)
            output, state = lstm_cell(i, output, state)
            if drop_out:
                output = tf.nn.dropout(output,keep_prob=keep_prob)
            outputs.append(output)

        with tf.control_dependencies([saved_output.assign(output),saved_state.assign(state)]):
            # logits = tf.nn.xw_plus_b(tf.concat(outputs,0), w, b)
            # logits = tf.nn.tanh(logits)
            # norm = tf.sqrt(tf.reduce_sum(tf.square(logits),1, keep_dims = True))
            # predictions = logits / norm
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.concat(train_labels, 0), logits=logits))
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=tf.transpose(w),biases=b,labels=tf.concat(train_labels,0),inputs=tf.concat(outputs,0),num_sampled=self.num_sampled, num_classes=self.vocabulary_size))
            # self.loss = tf.losses.cosine_distance(labels=tf.concat(train_labels, 0),predictions=predictions, dim = 1)

        global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(
        10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        self.optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
        self.train_prediction = tf.nn.softmax(tf.nn.xw_plus_b(tf.concat(outputs,0), w, b))



        self.sample_input = tf.placeholder(tf.float32, shape=[1, self.embedding_size])
        saved_sample_output = tf.Variable(tf.zeros([1, self.num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, self.num_nodes]))
        self.reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, self.num_nodes])),saved_sample_state.assign(tf.zeros([1, self.num_nodes])))
        sample_output, sample_state = lstm_cell(self.sample_input, saved_sample_output, saved_sample_state)
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                saved_sample_state.assign(sample_state)]):
            self.sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))






    def random_distribution(self):
        b = np.random.uniform(0.0, 1.0, size = [1,self.embedding_size])
        return b / np.sum(b,1)[:,None]



    def session_run(self, session, batches, labels):
        feed_dict = dict()
        for i in range(self.num_unrollings + 1):
            feed_dict[self.train_data[i]] = batches[i]
            feed_dict[self.train_label[i]] = labels[i]

        return session.run([self.optimizer, self.loss, self.train_prediction, self.learning_rate], feed_dict = feed_dict)

def label_to_one_hot(labels, vocabulary_size):
    label_one_hot = np.zeros(shape=(labels.shape[0], vocabulary_size), dtype=np.float)
    for index in range(labels.shape[0]):
        label_one_hot[index, labels[index,0]] = 1.0
    return label_one_hot

def logprob(predictions, labels):
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1

def LSTM_Session_Run(train_lstmbatcher, valid_lstmbatcher, num_nodes, num_steps, sample_frequency,num_sampled, drop_out = False, keep_prob = 0.5):
    graph = tf.Graph()
    train_batch_size = train_lstmbatcher.batch_size
    embeddings_size = train_lstmbatcher.embeddings_size
    num_unrollings = train_lstmbatcher.num_unrollings
    vocabulary_size = train_lstmbatcher.dataset_embeddings.get_grams_len()

    with graph.as_default():
        lstmgen = LSTMGenerater(batch_size=train_batch_size,embeddings_size=embeddings_size,num_unrollings=num_unrollings, num_nodes=num_nodes,drop_out = drop_out, keep_prob=keep_prob,vocabulary_size=vocabulary_size,num_sampled=num_sampled)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        mean_loss = 0
        for step in range(num_steps):
            batches, labels = train_lstmbatcher.next()
            _, l, prediction, lr = lstmgen.session_run(session,batches,labels)
            mean_loss += l
            if step % sample_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / sample_frequency
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0
                # labels = np.concatenate(list(batches)[1:])
                label_no = np.concatenate(list(labels)[1:])
                label_one_hot = label_to_one_hot(label_no, vocabulary_size)
                print('Minibatch perplexity: %.2f' % float(np.exp(logprob(prediction, label_one_hot))))

                if step % (sample_frequency * 10) == 0:
                    print ('=' * 80)
                    for _ in range(5):
                        random_embeddings = lstmgen.random_distribution()
                        # print random_embeddings.shape
                        random_embeddings_id = train_lstmbatcher.dataset_embeddings.get_id_from_predict_embedding(random_embeddings)
                        feed = train_lstmbatcher.dataset_embeddings.get_embedding_from_id(random_embeddings_id)
                        sentence = train_lstmbatcher.dataset_embeddings.get_gram_from_id(random_embeddings_id)
                        lstmgen.reset_sample_state.run()
                        for _ in range(79):
                            prediction = lstmgen.sample_prediction.eval({lstmgen.sample_input:feed})
                            # print prediction.shape
                            prediction_id = sample_distribution(prediction[0])

                            feed = train_lstmbatcher.dataset_embeddings.get_embedding_from_id(prediction_id)
                            sentence += train_lstmbatcher.dataset_embeddings.get_gram_from_id(prediction_id)
                        print (sentence)
                    print ('=' * 80)
                #Measure validation set perplexity
                lstmgen.reset_sample_state.run()
                valid_logprob = 0
                for _ in range(valid_lstmbatcher.text_size):
                    valid,valid_label = valid_lstmbatcher.next()
                    valid_label_no = np.concatenate(list(valid_label)[1:])
                    predictions = lstmgen.sample_prediction.eval({lstmgen.sample_input: valid[0]})
                    valid_label_one_hot = label_to_one_hot(valid_label_no,vocabulary_size)
                    valid_logprob = valid_logprob + logprob(predictions, valid_label_one_hot)
                print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_lstmbatcher.text_size)))














