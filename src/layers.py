"""
Author: Saravanakumar Shanmugam Sakthivadivel, 2018
Github: https://github.com/codewithsk/graph-cnn.mxnet

The Ohio State University
Graph Convolutional Network

File: layers.py
Description: Defines the graph convolution
             described in https://arxiv.org/abs/1609.02907
"""

import numpy as np #pylint: disable=import-error

import mxnet as mx # pylint: disable=import-error
from mxnet import nd # pylint: disable=import-error
from mxnet.gluon import Block # pylint: disable=import-error

class GraphConvolution(Block): # pylint: disable=too-few-public-methods
    """
    Defines the Graph Convolution layer as a Gluon Block.
    Inherits from Gluon Block
    """
    def __init__(self, in_features, out_features, bias=True, **kwargs):
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
            if bias:
                self.bias = self.params.get('bias',
                                            shape=(out_features,),
                                            dtype=np.float64,
                                            init=mx.init.One()
                                            )
            else:
                self.bias = self.params.get('bias',
                                            shape=(out_features,),
                                            dtype=np.float64,
                                            init=mx.init.Zero()
                                            )

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
        return output + self.bias.data()
