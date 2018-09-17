import tensorflow as tf
import numpy as np
import cPickle as pickle

import utils.layers as layers
import utils.operations as op

class Net(object):
    '''Add positional encoding(initializer lambda is 0),
       cross-attention, cnn integrated and grad clip by value.

    Attributes:
        conf: a configuration paramaters dict
        word_embedding_init: a 2-d array with shape [vocab_size+1, emb_size]
    '''
    def __init__(self, conf):
        self._graph = tf.Graph()
        self._conf = conf

        if self._conf['word_emb_init'] is not None:
            print('loading word emb init')
            self._word_embedding_init = pickle.load(open(self._conf['word_emb_init'], 'rb'))
        else:
            self._word_embedding_init = None

    def build_graph(self):
        with self._graph.as_default():
            rand_seed = self._conf['rand_seed']
            tf.set_random_seed(rand_seed)

            #word embedding
            if self._word_embedding_init is not None:
                word_embedding_initializer = tf.constant_initializer(self._word_embedding_init)
            else:
                word_embedding_initializer = tf.random_normal_initializer(stddev=0.1)

            self._word_embedding = tf.get_variable(
                name='word_embedding',
                shape=[self._conf['vocab_size']+1, self._conf['emb_size']],
                dtype=tf.float32,
                initializer=word_embedding_initializer)


            #define placehloders
            self.turns = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_num"], self._conf["max_turn_len"]])

            self.tt_turns_len = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"]])

            self.every_turn_len = tf.placeholder(
                tf.int32,
                shape=[self._conf["batch_size"], self._conf["max_turn_num"]])
    
            self.response = tf.placeholder(
                tf.int32, 
                shape=[self._conf["batch_size"], self._conf["max_turn_len"]])

            self.response_len = tf.placeholder(
                tf.int32, 
                shape=[self._conf["batch_size"]])

            self.label = tf.placeholder(
                tf.float32, 
                shape=[self._conf["batch_size"]])


            #define operations
            #response part
            Hr = tf.nn.embedding_lookup(self._word_embedding, self.response)
            #Hr_stack = [Hr]

            if self._conf['is_positional'] and self._conf['stack_num'] > 0:
                with tf.variable_scope('positional'):
                    Hr = op.positional_encoding_vector(Hr, max_timescale=10)
            Hr_stack = [Hr]

            for index in range(self._conf['stack_num']):
                with tf.variable_scope('self_stack_' + str(index)):
                    Hr = layers.block(
                        Hr, Hr, Hr, 
                        Q_lengths=self.response_len, K_lengths=self.response_len)
                    Hr_stack.append(Hr)

            Hr_stack = tf.stack(Hr_stack, axis=-1)


            #context part
            #a list of length max_turn_num, every element is a tensor with shape [batch, max_turn_len]
            list_turn_t = tf.unstack(self.turns, axis=1) 
            list_turn_length = tf.unstack(self.every_turn_len, axis=1)
            
            sim_turns = []
            #for every turn_t calculate matching vector
            for turn_t, t_turn_length in zip(list_turn_t, list_turn_length):
                Hu = tf.nn.embedding_lookup(self._word_embedding, turn_t) #[batch, max_turn_len, emb_size]
                #Hu_stack = [Hu]

                if self._conf['is_positional'] and self._conf['stack_num'] > 0:
                    with tf.variable_scope('positional', reuse=True):
                        Hu = op.positional_encoding_vector(Hu, max_timescale=10)
                Hu_stack = [Hu]


                for index in range(self._conf['stack_num']):

                    with tf.variable_scope('self_stack_' + str(index), reuse=True):
                        Hu = layers.block(
                            Hu, Hu, Hu,
                            Q_lengths=t_turn_length, K_lengths=t_turn_length)

                        Hu_stack.append(Hu)


                Hu_stack = tf.stack(Hu_stack, axis=-1)
                #print('Hu_stack shape: %s' %Hu_stack.shape)
                
                #calculate similarity matrix
                with tf.variable_scope('similarity'):
                    # sim shape [batch, max_turn_len, max_turn_len, 2*stack_num+1]
                    # divide sqrt(200) to prevent gradient explosion
                    sim = tf.einsum('biks,bjks->bijs', Hu_stack, Hr_stack) / tf.sqrt(200.0)

                sim_turns.append(sim)


            #cnn and aggregation
            sim = tf.stack(sim_turns, axis=1)
            print('sim shape: %s' %sim.shape)
            with tf.variable_scope('cnn_aggregation'):
                final_info = layers.CNN_3d(sim, 32, 16)
                #for douban
                #final_info = layers.CNN_3d(sim, 16, 16)


            #loss and train
            with tf.variable_scope('loss'):
                self.loss, self.logits = layers.loss(final_info, self.label)

                self.global_step = tf.Variable(0, trainable=False)
                initial_learning_rate = self._conf['learning_rate']
                self.learning_rate = tf.train.exponential_decay(
                    initial_learning_rate,
                    global_step=self.global_step,
                    decay_steps=400,
                    decay_rate=0.9,
                    staircase=True)

                Optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.optimizer = Optimizer.minimize(self.loss)

                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver(max_to_keep = self._conf["max_to_keep"])
                self.all_variables = tf.global_variables() 
                self.all_operations = self._graph.get_operations()
                self.grads_and_vars = Optimizer.compute_gradients(self.loss)

                for grad, var in self.grads_and_vars:
                    if grad is None:
                        print var

                self.capped_gvs = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.grads_and_vars]
                self.g_updates = Optimizer.apply_gradients(
                    self.capped_gvs,
                    global_step=self.global_step)
    
        return self._graph

