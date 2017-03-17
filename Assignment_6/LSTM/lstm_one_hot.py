__author__ = 'pandaczm'

import tensorflow as tf
import numpy as np
import random
import string



class LSTMGenerater(object):
    def __init__(self, embeddings_size, batch_size, num_unrollings, num_nodes, num_sampled):

        self.embedding_size = embeddings_size
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

        w = tf.Variable(tf.truncated_normal([self.num_nodes, self.embedding_size], -0.1, 0.1))
        b = tf.Variable(tf.zeros([self.embedding_size]))



        self.train_data = list()
        for _ in range(self.num_unrollings + 1):
            self.train_data.append(tf.placeholder(tf.float32, shape = [self.batch_size, self.embedding_size]))
        train_inputs = self.train_data[: self.num_unrollings]
        # train_labels = self.train_data[1:]
        self.train_label = list()
        for _ in range(self.num_unrollings + 1):
            self.train_label.append(tf.placeholder(tf.float32, shape = [self.batch_size, 1]))
        train_labels = self.train_label[1:]
        train_labels = self.train_data[1:]

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
            output, state = lstm_cell(i, output, state)
            outputs.append(output)

        with tf.control_dependencies([saved_output.assign(output),saved_state.assign(state)]):
            logits = tf.nn.xw_plus_b(tf.concat(outputs,0), w, b)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = tf.concat(train_labels, 0), logits=logits))
            # self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=tf.transpose(w),biases=b,labels=tf.concat(train_labels,0),inputs=tf.concat(outputs,0),num_sampled=self.num_sampled, num_classes=self.embedding_size))

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

def logprob(predictions, labels):
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def LSTM_Session_Run(train_lstmbatcher, valid_lstmbatcher, num_nodes, num_steps,sample_frequency, num_sampeled):
    graph = tf.Graph()
    train_batch_size = train_lstmbatcher.batch_size
    embeddings_size = train_lstmbatcher.embeddings_size
    num_unrollings = train_lstmbatcher.num_unrollings

    with graph.as_default():
        lstmgen = LSTMGenerater(batch_size=train_batch_size,embeddings_size=embeddings_size,num_unrollings=num_unrollings, num_nodes=num_nodes, num_sampled=num_sampeled)

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        mean_loss = 0
        for step in range(num_steps):
            batches, labels= train_lstmbatcher.next()
            _, l, prediction, lr = lstmgen.session_run(session,batches,labels)
            mean_loss += l
            if step % sample_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / sample_frequency
                print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
                mean_loss = 0
                labels = np.concatenate(list(batches)[1:])
                print('Minibatch perplexity: %.2f' % float(np.exp(logprob(prediction, labels))))

                if step % (sample_frequency * 10) == 0:
                    print ('=' * 80)
                    for _ in range(5):
                        random_embeddings = lstmgen.random_distribution()
                        random_embeddings_id = train_lstmbatcher.dataset_embeddings.get_id_from_predict_embedding(random_embeddings)
                        feed = train_lstmbatcher.dataset_embeddings.get_embedding_from_id(random_embeddings_id)
                        sentence = train_lstmbatcher.dataset_embeddings.get_gram_from_id(random_embeddings_id)
                        lstmgen.reset_sample_state.run()
                        for _ in range(79):
                            prediction = lstmgen.sample_prediction.eval({lstmgen.sample_input:feed})
                            # print prediction.shape
                            prediction_id = train_lstmbatcher.dataset_embeddings.get_id_from_predict_embedding(prediction)
                            feed = train_lstmbatcher.dataset_embeddings.get_embedding_from_id(prediction_id)
                            sentence += train_lstmbatcher.dataset_embeddings.get_gram_from_id(prediction_id)
                        print (sentence)
                    print ('=' * 80)
                    lstmgen.reset_sample_state.run()
                valid_logprob = 0
                for _ in range(valid_lstmbatcher.text_size):
                    b,_ = valid_lstmbatcher.next()
                    predictions = lstmgen.sample_prediction.eval({lstmgen.sample_input: b[0]})
                    valid_logprob = valid_logprob + logprob(predictions, b[1])
                print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_lstmbatcher.text_size)))