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
    pool1 = mx.symbol.Pooling(
        data=relu1_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool1")
    # group 2
    conv2_11 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 1), pad=(1, 1), num_filter=128, name="conv2_11")
    relu2_11 = mx.symbol.Activation(data=conv2_11, act_type="relu", name="relu2_11")
    conv2_1 = mx.symbol.Convolution(
        data=relu2_11, kernel=(1, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    pool2 = mx.symbol.Pooling(
        data=relu2_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool2")
    # group 3
    conv3_11 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 1), pad=(1, 1), num_filter=256, name="conv3_11")
    relu3_11 = mx.symbol.Activation(data=conv3_11, act_type="relu", name="relu3_11")
    conv3_1 = mx.symbol.Convolution(
        data=relu3_11, kernel=(1, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_21 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 1), pad=(1, 1), num_filter=256, name="conv3_21")
    relu3_21 = mx.symbol.Activation(data=conv3_21, act_type="relu", name="relu3_21")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_21, kernel=(1, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    pool3 = mx.symbol.Pooling(
        data=relu3_2, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool3")
    # group 4
    conv4_11 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 1), num_filter=512, name="conv4_11")
    relu4_11 = mx.symbol.Activation(data=conv4_11, act_type="relu", name="relu4_11")
    conv4_1 = mx.symbol.Convolution(
        data=relu4_11, kernel=(1, 3), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_21 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 1), pad=(1, 1), num_filter=512, name="conv4_21")
    relu4_21 = mx.symbol.Activation(data=conv4_21, act_type="relu", name="relu4_21")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_21, kernel=(1, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    pool4 = mx.symbol.Pooling(
        data=relu4_1, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool4")
    # group 5
    conv5_11 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 1), pad=(1, 1), num_filter=512, name="conv5_11")
    relu5_11 = mx.symbol.Activation(data=conv5_11, act_type="relu", name="relu5_11")
    conv5_1 = mx.symbol.Convolution(
        data=relu5_11, kernel=(1, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_21 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 1), pad=(1, 1), num_filter=512, name="conv5_21")
    relu5_21 = mx.symbol.Activation(data=conv5_21, act_type="relu", name="relu5_21")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_21, kernel=(1, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
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
