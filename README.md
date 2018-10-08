<p align="center">
  <img src="assets/mxnet_logo_2.png"/>
</p>

MXNet implementation of Graph Convolutional Neural Networks detailed in [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)


[Tensorflow Implementation (Original)](https://github.com/tkipf/gcn)
[Author's PyTorch Implementation](https://github.com/tkipf/pygcn)
[Pytorch Implementation by Bumsoo Kim](https://github.com/meliketoy/graph-cnn.pytorch)

## Graph Convolutional Networks
Graph Convolutions are best explained in [this amazing blogpost](https://tkipf.github.io/graph-convolutional-networks/) by [Thomas Kipf](https://twitter.com/thomaskipf?lang=en)
the author of the paper [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)

## Requirements
- Python
- MXnet
- Numpy

## Usage
```bash
python train.py
```

## Notes/Observations:
1. Unable to reproduce results from the paper with dropout of 0.5. Dropout=0 gives results similar to paper

## TODO:
1. NetworkX integration for loading data
2. Add options to pick loss function
3. Support datasets other than CORA
4. Option to save model
5. Add visualization support of loss parameters

