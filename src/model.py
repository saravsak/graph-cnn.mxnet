import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block
import mxnet.ndarray as F
from layers import GraphConvolution

class GCN(Block):
    def __init__(self, nfeat, nhid, nclass, dropout, **kwargs):
        super(GCN, self).__init__(**kwargs)
        with self.name_scope():
            self.gc1 = GraphConvolution(nfeat, nhid)
            self.gc2 = GraphConvolution(nhid, nclass)
            self.dropout = dropout

    def forward(self, x, adj, training=True):
        x = F.relu(self.gc1(x, adj))
        #x = gluon.nn.Dropout(x, self.dropout)
        x = self.gc2(x, adj)
        return F.log_softmax(x, axis=1)
