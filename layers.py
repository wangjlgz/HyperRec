import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):

    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs



class HGNN_conv(Layer):
    """Basic hypergraph convolution layer."""
    def __init__(self, input_dim, output_dim, adj, n_hyper, dropout=0., act=tf.nn.relu, **kwargs):
        super(HGNN_conv, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            k = 0
            self.vars['weights_%d' %k] = weight_variable_glorot(input_dim, output_dim, name="weights_%d" %k)
            for k in range(1, n_hyper+1):
                self.vars['weights_%d' %k] = weight_variable_glorot(output_dim, output_dim, name="weights_%d" %k)
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.n_hyper = n_hyper

    def _call(self, inputs):
        x = inputs
        # first layer - no dropout
        k = 0
        x = tf.matmul(x, self.vars['weights_%d' %k])
        y = tf.sparse_tensor_dense_matmul(self.adj['E'], x)
        x = tf.sparse_tensor_dense_matmul(self.adj['G'], x)

        x = self.act(x)

        for k in range(1, self.n_hyper):
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights_%d' %k])
            x = tf.sparse_tensor_dense_matmul(self.adj['G'], x)
            x = self.act(x)
        k = self.n_hyper
        x1 = tf.nn.dropout(x, 1-self.dropout)
        y = tf.sparse_tensor_dense_matmul(self.adj['E'], x1)
        y = self.act(y)
        return x,y

