import numpy as np
import tensorflow as tf

class GRUTensorflow(object):
    def __init__(self, word_dim, hidden_dim=128, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # 初始化网络参数
        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((6, hidden_dim))
        c = np.zeros(word_dim)

        # 共享变量
        self.E = tf.Variable(E,dtype=tf.float32,name='E')
        self.U = tf.Variable(U,dtype=tf.float32,name='U')
        self.W = tf.Variable(W,dtype=tf.float32,name='W')
        self.V = tf.Variable(V,dtype=tf.float32,name='V')
        self.b = tf.Variable(b,dtype=tf.float32,name='b')
        self.c = tf.Variable(c,dtype=tf.float32,name='c')

        #随机梯度下降:初始化参数
        self.mE = tf.Variable(np.zeros(E.shape), dtype=tf.float32, name='mE')
        self.mU = tf.Variable(np.zeros(U.shape), dtype=tf.float32, name='mU')
        self.mW = tf.Variable(np.zeros(W.shape), dtype=tf.float32, name='mW')
        self.mV = tf.Variable(np.zeros(V.shape), dtype=tf.float32, name='mV')
        self.mb = tf.Variable(np.zeros(b.shape), dtype=tf.float32, name='mb')
        self.mc = tf.Variable(np.zeros(c.shape), dtype=tf.float32, name='mc')

    def build_graph(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c

        x = tf.placeholder(tf.int32)
        y = tf.placeholder(tf.int32)

        def forward_prop_step(x_t, s_t1_prev, s_t2_prev):
            # This is how we calculated the hidden state in a simple RNN. No longer!
            # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))

            # Word embedding layer
            x_e = E[:, x_t]
            num_layers = 2
            dropout = tf.placeholder(tf.float32)
            # GRU Layer
            gru_cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
            gru_cell = tf.nn.rnn_cell.DropoutWrapper(gru_cell, output_keep_prob=dropout)
            gru_cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell] * num_layers)


            z_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            r_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            c_t1 = T.tanh(U[2].dot(x_e) + W[2].dot(s_t1_prev * r_t1) + b[2])
            s_t1 = (T.ones_like(z_t1) - z_t1) * c_t1 + z_t1 * s_t1_prev

            # GRU Layer 2
            z_t2 = T.nnet.hard_sigmoid(U[3].dot(s_t1) + W[3].dot(s_t2_prev) + b[3])
            r_t2 = T.nnet.hard_sigmoid(U[4].dot(s_t1) + W[4].dot(s_t2_prev) + b[4])
            c_t2 = T.tanh(U[5].dot(s_t1) + W[5].dot(s_t2_prev * r_t2) + b[5])
            s_t2 = (T.ones_like(z_t2) - z_t2) * c_t2 + z_t2 * s_t2_prev

            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(s_t2) + c)[0]

            return [o_t, s_t1, s_t2]