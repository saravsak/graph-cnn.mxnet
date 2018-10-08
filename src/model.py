"""
Author: Saravanakumar Shanmugam Sakthivadivel, 2018
Github: https://github.com/codewithsk/graph-cnn.mxnet

The Ohio State University
Graph Convolutional Network

File: model.py
Description: Defines the graph convolution network model
             using the Graph Convolution Layer described
             in https://arxiv.org/abs/1609.02907 as
             defined in layers.py
"""
import mxnet.ndarray as F
from mxnet.gluon import Block

from layers import GraphConvolution

class GCN(Block):
    """
    Defines a two layer Graph Convolutional Network.
    Inherits from Gluon Block
    """
    def __init__(self, nfeat, nhid, nclass, dropout, bias=True, **kwargs): #pylint: disable=too-many-arguments
        """
        Constructor for Graph Convolution Network.

        Params
        ======
            nfeat: int
                   Number of input features
            nhid: int
                   Number of neurons in the hidden layers
            nclass: int
                   Number of output layers
            dropout: float (0~1)
                   Dropout probability
            bias: bool
                   Whether bias should be included or not
        Returns:
        =======
            None
        """
        super(GCN, self).__init__(**kwargs)
        with self.name_scope():
            self.gc1 = GraphConvolution(nfeat, nhid, bias)
            self.gc2 = GraphConvolution(nhid, nclass, bias)
            self.dropout = dropout

    def forward(self, x, adj): # pylint: disable=arguments-differ
        """
        Forward pass for Graph Convolution Network.

        Params
        ======
            x: NDArray
                Input embeddings of size (batch_size, number of input features)
            adj: NDArray
                Adjacency matrix (number of input_features, number of input_features)
        Returns:
        =======
            NDArray
        """
        x = F.relu(self.gc1(x, adj)) # pylint: disable=no-member
        if self.dropout > 0:
            x = F.Dropout(x, self.dropout) # pylint: disable=no-member
        x = self.gc2(x, adj)
        return F.log_softmax(x, axis=1) # pylint: disable=no-member
