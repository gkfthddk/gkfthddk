"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
import mxnet as mx

def get_symbol(num_classes, **kwargs):
    ## define VGG11
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_11 = mx.symbol.Convolution(data=data, kernel=(3, 1), pad=(1, 0), num_filter=64, name="conv1_11")
    relu1_11 = mx.symbol.Activation(data=conv1_11, act_type="relu", name="relu1_11")
    conv1_1 = mx.symbol.Convolution(data=relu1_11, kernel=(1, 3), pad=(0, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_12 = mx.symbol.Convolution(data=relu1_1, kernel=(3, 1), pad=(1, 0), num_filter=64, name="conv1_12")
    relu1_12 = mx.symbol.Activation(data=conv1_12, act_type="relu", name="relu1_12")
    conv1_2 = mx.symbol.Convolution(data=relu1_12, kernel=(1, 3), pad=(0, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    
    flatten = mx.symbol.Flatten(data=pool1, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # output
    fc8 = mx.symbol.FullyConnected(data=drop7, num_hidden=num_classes, name="fc8")
    softmax = mx.symbol.SoftmaxOutput(data=fc8, name='softmax')
    return softmax
