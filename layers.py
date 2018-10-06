import mxnet as mx
import numpy as np
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, Block

class GraphConvolution(Block):
    def __init__(self, in_features, out_features, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with self.name_scope():
            self.in_features = in_features
            self.out_features = out_features
            self.weight = self.params.get(
                'weight',
                init=mx.init.Xavier(magnitude=2.24),
                shape=(in_features, out_features)
            )
            # TODO: Make bias optional
            self.bias = self.params.get('bias', shape=(out_features,))

    def forward(self, inp, adj):
        with inp.context:
            support = nd.dot(inp, self.weight.data()) + self.bias.data()
            output = nd.dot(adj, support)
            # TODO: Make bias optional
            return output + self.bias.data()
