"""
Author: Saravanakumar Shanmugam Sakthivadivel, 2018
Github: https://github.com/codewithsk/graph-cnn.mxnet

The Ohio State University
Graph Convolutional Network

File: layers.py
Description: Defines the graph convolution
             described in https://arxiv.org/abs/1609.02907
"""

import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block

class GraphConvolution(Block):
    """
    Defines the Graph Convolution layer as a Gluon Block.
    Inherits from Gluon Block
    """
    def __init__(self, in_features, out_features, **kwargs):
        """
        Constructor for Graph Convolution layer.

        Params
        ======
            in_features: int
                         Number of input features for this layer
            out_features: int
                          Number of output features for this layer

        Returns:
        =======
            None
        """
        super(GraphConvolution, self).__init__(**kwargs)
        with self.name_scope():
            self.in_features = in_features
            self.out_features = out_features
            self.weight = self.params.get(
                'weight',
                init=mx.init.Xavier(magnitude=2.24),
                shape=(in_features, out_features),
                dtype=np.float64
            )
            # TODO: Make bias optional
            self.bias = self.params.get('bias', shape=(out_features,), dtype=np.float64)

    def forward(self, inp, adj):
        """
        Forward pass for Graph Convolution Layer.

        Params:
        ======
            inp: NDArray
                 Input vector of size (batch_size,in_features)
            adj: NDArray
                 Adjacency matrix of size (in_features, in_features)

        Returns:
        =======
            NDArray
        """
        support = nd.dot(inp, self.weight.data()) + self.bias.data()
        output = nd.dot(adj, support)
        # TODO: Make bias optional
        return output + self.bias.data()
