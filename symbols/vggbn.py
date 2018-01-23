"""References:

Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for
large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""
import mxnet as mx

def get_symbol(num_classes, **kwargs):
    ## define VGG11
    data = mx.symbol.Variable(name="data")
    # group 1
    conv1_1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    bnor1_1 = mx.symbol.BatchNorm(conv1_1,fix_gamma=False, eps=2e-5, momentum=0.9,name="bnor1_1")
    relu1_1 = mx.symbol.Activation(data=bnor1_1, act_type="relu", name="relu1_1")
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    bnor2_1 = mx.symbol.BatchNorm(conv2_1,fix_gamma=False, eps=2e-5, momentum=0.9,name="bnor1_2")
    relu2_1 = mx.symbol.Activation(data=bnor2_1, act_type="relu", name="relu2_1")
    pool2 = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    bnor3_1 = mx.symbol.BatchNorm(conv3_1,fix_gamma=False, eps=2e-5, momentum=0.9,name="bnor3_1")
    relu3_1 = mx.symbol.Activation(data=bnor3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    bnor3_2 = mx.symbol.BatchNorm(conv3_2,fix_gamma=False, eps=2e-5, momentum=0.9,name="bnor3_2")
    relu3_2 = mx.symbol.Activation(data=bnor3_2, act_type="relu", name="relu3_2")
    pool3 = mx.symbol.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), num_filter=512, name="conv4_1")
    bnor4_1 = mx.symbol.BatchNorm(conv4_1,fix_gamma=False, eps=2e-5, momentum=0.9,name="bnor4_1")
    relu4_1 = mx.symbol.Activation(data=bnor4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    bnor4_2 = mx.symbol.BatchNorm(conv4_2,fix_gamma=False, eps=2e-5, momentum=0.9,name="bnor4_2")
    relu4_2 = mx.symbol.Activation(data=bnor4_2, act_type="relu", name="relu4_2")
    pool4 = mx.symbol.Pooling(
        data=relu4_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    bnor5_1 = mx.symbol.BatchNorm(conv5_1,fix_gamma=False, eps=2e-5, momentum=0.9,name="bnor5_1")
    relu5_1 = mx.symbol.Activation(data=bnor5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    bnor5_2 = mx.symbol.BatchNorm(conv5_2,fix_gamma=False, eps=2e-5, momentum=0.9,name="bnor5_2")
    relu5_2 = mx.symbol.Activation(data=bnor5_2, act_type="relu", name="relu5_2")
    pool5 = mx.symbol.Pooling(
        data=relu5_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # group 6
    flatten = mx.symbol.Flatten(data=pool4, name="flatten")
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