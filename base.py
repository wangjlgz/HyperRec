# Part of this file is derived from 
# https://github.com/DeepGraphLearning/RecommenderSystems

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


def normalize(inputs, 
              epsilon = 1e-8,
              scope="ln",
              reuse=None):

    '''Applies layer normalization.
    
    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''

    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs


def multihead_attention(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_keep_prob=1.0,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None,
                        with_qk=False):
    '''
    Applies multihead attention.
    
    Args:
        queries: A 3d tensor with shape of [N, T_q, C_q].
        keys: A 3d tensor with shape of [N, T_k, C_k].
        num_units: A scalar. Attention size.
        dropout_rate: A floating point number.
        is_training: Boolean. Controller of mechanism for dropout.
        causality: Boolean. If true, units that reference the future are masked. 
        num_heads: An int. Number of heads.
        scope: Optional scope for `variable_scope`.
        reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.
        
    Returns
        A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        # Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # (N, T_q, C)
        # K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        # V = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # (N, T_k, C)
        Q = tf.layers.dense(queries, num_units, activation=None) # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=None) # (N, T_k, C)
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # (h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) # (h*N, T_q, T_k)
        
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k) if key==0 set mask =-inf
  
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :]) # (T_q, T_k)
            try:
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            except:
                tril = tf.linalg.LinearOperatorTriL(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1]) # (h*N, T_q, T_k)
   
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (h*N, T_q, T_k)
         
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1]) # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (N, T_q, C)
          
        # Dropouts
        outputs = tf.nn.dropout(outputs, keep_prob=dropout_keep_prob)
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # (N, T_q, C)
              
        # Residual connection
        outputs += queries
              
        # Normalize
        #outputs = normalize(outputs) # (N, T_q, C)
 
    if with_qk: 
        return Q, K
    else: 
        return outputs


def feedforward(inputs, 
                num_units=[2048, 512],
                scope="multihead_attention", 
                dropout_keep_prob=1.0,
                reuse=None):
    '''
    Point-wise feed forward net.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.nn.dropout(outputs, keep_prob=dropout_keep_prob)
        #outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        outputs = tf.nn.dropout(outputs, keep_prob=dropout_keep_prob)
        #outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        #outputs = normalize(outputs)
    
    return outputs


class TransformerNet(object):
    def __init__(self, num_units, num_blocks, num_heads, maxlen, dropout_rate, pos_fixed, reuse, l2_reg=0.0):
        self.num_units = num_units
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.dropout_keep_prob = 1. - dropout_rate
        self.position_encoding_matrix = None
        self.pos_fixed = pos_fixed
        self.l2_reg = l2_reg
        self.reuse = reuse
        #self.position_encoding = position_encoding(self.maxlen, self.num_units) # (maxlen, num_units)

    def position_embedding(self, inputs, maxlen, num_units, l2_reg=0.0, scope="pos_embedding", reuse=None, zero_pad=False):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            pos_embedding_table = tf.get_variable('pos_embedding_table', dtype=tf.float32, shape=[maxlen, num_units], regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            if zero_pad:
                pos_embedding_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                          pos_embedding_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(pos_embedding_table, inputs)
        return outputs

    def get_position_encoding(self, inputs, scope="pos_embedding/", reuse=None, dtype=tf.float32):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self.position_encoding_matrix is None:
                encoded_vec = np.array([pos/np.power(10000, 2*i/self.num_units) for pos in range(self.maxlen) for i in range(self.num_units)])
                encoded_vec[::2] = np.sin(encoded_vec[::2])
                encoded_vec[1::2] = np.cos(encoded_vec[1::2])
                encoded_vec = tf.convert_to_tensor(encoded_vec.reshape([self.maxlen, self.num_units]), dtype=dtype)
                self.position_encoding_matrix = encoded_vec # (maxlen, num_units)
                
            N = tf.shape(inputs)[0]
            T = tf.shape(inputs)[1]
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) # (batch_size , len)
            position_encoding = tf.nn.embedding_lookup(self.position_encoding_matrix, position_ind) # (batch_size, len, num_units)
        return position_encoding


    def __call__(self, inputs, mask):
        '''
        Args:
        inputs: sequence embeddings (item_embeddings +  pos_embeddings) shape: (batch_size , maxlen, embedding_size)
        Return:
        Output sequences which has the same shape with inputs
        '''
        if self.pos_fixed: # use sin /cos positional embedding
            position_encoding = self.get_position_encoding(inputs, reuse = self.reuse) # (batch_size, len, num_units)
        else:
            position_encoding = self.position_embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(inputs)[1]), 0), [tf.shape(inputs)[0], 1]), self.maxlen, self.num_units, self.l2_reg, reuse = self.reuse)
        inputs += position_encoding
        inputs = tf.nn.dropout(inputs, keep_prob = self.dropout_keep_prob)
        inputs *= mask
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # Self-attention
                inputs = multihead_attention(queries=normalize(inputs),
                                               keys=inputs,
                                               num_units=self.num_units,
                                               num_heads=self.num_heads,
                                               dropout_keep_prob=self.dropout_keep_prob,
                                               causality=True,
                                               scope="self_attention",
                                               reuse = self.reuse)

                # Feed forward
                inputs = feedforward(normalize(inputs), num_units=[self.num_units, self.num_units],
                                       dropout_keep_prob=self.dropout_keep_prob, reuse = self.reuse)

                inputs *= mask
        outputs = normalize(inputs)  # (batch_size, maxlen, num_units)
        return outputs

