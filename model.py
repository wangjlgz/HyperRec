import tensorflow as tf
import numpy as np
import sys
import scipy.sparse as sp

from base import TransformerNet
from layers import HGNN_conv

def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


class NeuralSeqRecommender(object):
    def __init__(self, args, n_items, n_users, subgraphs_G, reversed_subgraphs_mapping_i, reversed_subgraphs_mapping_u, reversed_subgraphs_mapping_last_i, sorted_time, n_hyper = 2):
        self.args = args
        self.n_items = n_items
        self.n_users = n_users
        self.sorted_time = sorted_time
        self.n_hyper = n_hyper      #number of hypergraph layers

        self.reversed_subgraphs_mapping_i = reversed_subgraphs_mapping_i   #for the original self-attention model
        self.reversed_subgraphs_mapping_u = reversed_subgraphs_mapping_u
        self.reversed_subgraphs_mapping_last_i = reversed_subgraphs_mapping_last_i

        self.subgraphs_G = {}
        for i in sorted_time:
            self.subgraphs_G[i] = {}
            self.subgraphs_G[i]['G'] = _convert_sp_mat_to_sp_tensor(subgraphs_G[i]['G'])
            self.subgraphs_G[i]['E'] = _convert_sp_mat_to_sp_tensor(subgraphs_G[i]['E'])


        self._build()
        self.build_bpr_graph()

        self.saver = tf.train.Saver()



    def build_bpr_graph(self):
        self.triple_bpr = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        with tf.name_scope('con_bpr'):
            x_pos, x_neg = self.infer_bpr(self.triple_bpr)
            self.loss_bpr = tf.reduce_sum(tf.log(1 + tf.exp(-(x_pos - x_neg))))
        with tf.name_scope('training'):
            self.optimizer_bpr = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss_bpr)


    def infer_bpr(self, triple_bpr):
        with tf.name_scope('lookup_bpr'):
            bpr_user = tf.nn.embedding_lookup(self.user_embedding, triple_bpr[:, 0])
            bpr_pos = tf.nn.embedding_lookup(self.item_embedding, triple_bpr[:, 1])
            bpr_neg = tf.nn.embedding_lookup(self.item_embedding, triple_bpr[:, 2])

        with tf.name_scope('cal_bpr'):
            x_pos = tf.reduce_sum(tf.multiply(bpr_user, bpr_pos), axis=1)
            x_neg = tf.reduce_sum(tf.multiply(bpr_user, bpr_neg), axis=1)

        return x_pos, x_neg


    def _init_weights(self):
    	self.all_weights = {}

    	initializer = tf.contrib.layers.xavier_initializer()

    	for i in self.sorted_time:
    		for n in range(self.n_hyper):
        		self.all_weights['W_'+str(i)+'_'+str(n)] = tf.Variable(initializer([self.args.emsize, self.args.emsize]), name='W_'+str(i)+'_'+str(n))

    def _build(self):

        self.inp = tf.placeholder(tf.int32, shape=(None, None), name='inp')
        self.inp_ori = tf.placeholder(tf.int32, shape=(None, None), name='inp_ori')
        self.pos = tf.placeholder(tf.int32, shape=(None, None), name='pos')
        self.neg = tf.placeholder(tf.int32, shape=(None, self.args.neg_size), name='neg')
        self.pos_dy = tf.placeholder(tf.int32, shape=(None, None), name='pos_dy')
        self.neg_dy = tf.placeholder(tf.int32, shape=(None, self.args.neg_size), name='neg_dy')

        self.u_list = tf.placeholder(tf.int32, shape=(None), name='u_list')
        self.u_list_dy = tf.placeholder(tf.int32, shape=(None), name='u_list_dy')

        self.u_seq = tf.placeholder(tf.int32, shape=(None, None), name='u_seq')


        self.lr = tf.placeholder(tf.float32, shape=None, name='lr')
        self.dropout = tf.placeholder_with_default(0., shape=())
        self.dropout_graph = tf.placeholder_with_default(0., shape=())
        self.item_embedding = item_embedding = tf.get_variable('item_embedding', \
                                shape=(self.n_items, self.args.emsize), \
                                dtype=tf.float32, \
                                regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg), \
                                initializer=tf.contrib.layers.xavier_initializer())
        

        self.user_embedding = user_embedding =  tf.get_variable('user_embedding', \
                        shape=(self.n_users, self.args.emsize), \
                        dtype=tf.float32, \
                        regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg), \
                        initializer=tf.contrib.layers.xavier_initializer())

        self.item_embedding = tf.concat((tf.zeros(shape=[1, self.args.emsize]),
                                      item_embedding[1:, :]), 0)

        

        self.user_embedding = tf.concat((tf.zeros(shape=[1, self.args.emsize]),
                                      user_embedding[1:, :]), 0)


        emb_list = self.item_embedding
        emb_list_user = [self.user_embedding]

        for i in self.sorted_time:
            x1 = tf.nn.embedding_lookup(self.item_embedding, self.reversed_subgraphs_mapping_i[i])
            x2 = tf.nn.embedding_lookup(emb_list, self.reversed_subgraphs_mapping_last_i[i])

            stacked_features = tf.stack([x1,x2])


            stacked_features_transformed = tf.layers.dense(stacked_features, self.args.emsize, activation=tf.nn.tanh)
            stacked_features_score = tf.layers.dense(stacked_features_transformed, 1)
            stacked_features_score = tf.nn.softmax(stacked_features_score, 0)
            stacked_features_score = tf.nn.dropout(stacked_features_score, keep_prob=1. - self.dropout)

            xx = tf.reduce_sum(stacked_features_score*stacked_features, 0)


            nodes, edges = HGNN_conv(input_dim = self.args.emsize,
                                           output_dim = self.args.emsize,
                                           adj = self.subgraphs_G[i],
                                           act = tf.nn.relu,
                                           dropout = self.dropout_graph,
                                           n_hyper = self.n_hyper)(xx)

            emb_list = tf.concat([emb_list, nodes],0)
            emb_list_user.append(edges)


        emb_list_user = tf.concat(emb_list_user, 0)



        input_item1 = tf.nn.embedding_lookup(emb_list, self.inp)
        input_item1 = input_item1 * (self.args.emsize ** 0.5)



        input_user = tf.nn.embedding_lookup(emb_list_user, self.u_seq)
        input_user = input_user * (self.args.emsize ** 0.5)


        input_item2 = tf.nn.embedding_lookup(self.item_embedding, self.inp_ori)
        input_item2 = input_item2 * (self.args.emsize ** 0.5)

        stacked_features_input = tf.stack([input_item1, input_item2, input_user])


        stacked_features_transformed_input = tf.layers.dense(stacked_features_input, self.args.emsize, activation=tf.nn.tanh)
        stacked_features_score_input = tf.layers.dense(stacked_features_transformed_input, 1)
        stacked_features_score_input = tf.nn.softmax(stacked_features_score_input, 0)
        stacked_features_score_input = tf.nn.dropout(stacked_features_score_input, keep_prob=1. - self.dropout)

        input_item_all = tf.reduce_sum(stacked_features_score_input*stacked_features_input, 0)



        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inp_ori, 0)), -1)

        self.net = TransformerNet(self.args.emsize, self.args.num_blocks, self.args.num_heads, self.args.seq_len, dropout_rate=self.dropout, pos_fixed=self.args.pos_fixed,reuse=True)


        outputs = self.net(input_item_all, mask)

        

        ct_vec_last = outputs[:,-1,:]
        ct_vec_last = tf.reshape(ct_vec_last, (-1, self.args.emsize))   

        ct_vec = tf.reshape(outputs, (-1, self.args.emsize))
        outputs_shape = tf.shape(outputs)



        self.total_loss = 0.

        self.istarget = istarget = tf.reshape(tf.to_float(tf.not_equal(self.pos, 0)), [-1])

        _pos_emb = tf.nn.embedding_lookup(self.item_embedding, self.pos)
        pos_emb = tf.reshape(_pos_emb, (-1, self.args.emsize))
        _neg_emb = tf.nn.embedding_lookup(self.item_embedding, self.neg)
        neg_emb = tf.reshape(_neg_emb, (-1, self.args.neg_size, self.args.emsize))


        _pos_emb_dy = tf.nn.embedding_lookup(emb_list, self.pos_dy)
        pos_emb_dy = tf.reshape(_pos_emb_dy, (-1, self.args.emsize))
        _neg_emb_dy = tf.nn.embedding_lookup(emb_list, self.neg_dy)
        neg_emb_dy = tf.reshape(_neg_emb_dy, (-1, self.args.neg_size, self.args.emsize))


        pos_emb_joint = pos_emb + pos_emb_dy


        neg_emb_joint = neg_emb + neg_emb_dy
        
        
        temp_vec_neg = tf.tile(tf.expand_dims(ct_vec_last, [1]), [1, self.args.neg_size, 1]) 
        

        
        assert self.args.neg_size == 1

        pos_logit = tf.reduce_sum(ct_vec_last * pos_emb_joint, -1) 
        neg_logit = tf.squeeze(tf.reduce_sum(temp_vec_neg * neg_emb_joint, -1), 1) 
        loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos_logit) + 1e-24) * istarget - \
                    tf.log(1 - tf.sigmoid(neg_logit) + 1e-24) * istarget \
                ) / tf.reduce_sum(istarget)

               
        ct_vec_batch = tf.tile(ct_vec_last, [101, 1])
        self.test_item_batch = tf.placeholder(tf.int32, shape=(None, 101), name='test_item_batch')
        _test_item_emb_batch = tf.nn.embedding_lookup(self.item_embedding, self.test_item_batch)
        _test_item_emb_batch = tf.transpose(_test_item_emb_batch, perm=[1, 0, 2])
        test_item_emb_batch = tf.reshape(_test_item_emb_batch, (-1, self.args.emsize))

        self.test_item_batch_dy = tf.placeholder(tf.int32, shape=(None, 101), name='test_item_batch_dy')
        _test_item_emb_batch_dy = tf.nn.embedding_lookup(emb_list, self.test_item_batch_dy)
        _test_item_emb_batch_dy = tf.transpose(_test_item_emb_batch_dy, perm=[1, 0, 2])
        test_item_emb_batch_dy = tf.reshape(_test_item_emb_batch_dy, (-1, self.args.emsize))


        test_item_emb_batch_joint = test_item_emb_batch + test_item_emb_batch_dy

        self.test_logits_batch = tf.reduce_sum(ct_vec_batch*test_item_emb_batch_joint, -1) 
        self.test_logits_batch = tf.transpose(tf.reshape(self.test_logits_batch, [101, tf.shape(self.inp)[0]]))

               
        self.loss = loss
        self.total_loss += loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss += sum(reg_losses)

        optimizer = tf.train.AdamOptimizer(self.lr)
        gvs = optimizer.compute_gradients(self.total_loss)
        capped_gvs = map(lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -10., 10.), gv[1]], gvs)
        self.train_op = optimizer.apply_gradients(capped_gvs)        