import gzip
import os
from lcztools.config import get_global_config
import numpy as np
from functools import partial

# Note: Weight loading code taken from
# https://github.com/glinscott/leela-chess/blob/master/training/tf/net_to_model.py


LEELA_WEIGHTS_VERSION = '2'

def read_weights_file(filename):
    config = get_global_config()
    filename = config.get_weights_filename(filename)
    filename = os.path.expanduser(filename)

    if filename.endswith('.gz'):
        opener = partial(gzip.open, filename)
        filename_rest = filename[:-3]
    else:
        opener = partial(open, filename)
        filename_rest = filename
    
    if filename_rest.endswith('.pb') or filename_rest.endswith('.pb2'):
        return read_weights_file_pb2(opener)
    else:
        return read_weights_file_txt(opener)


def read_weights_file_txt(opener):
    with opener(filename, 'r') as f:
        version = f.readline().decode('ascii')
        if version != '{}\n'.format(LEELA_WEIGHTS_VERSION):
            raise ValueError("Invalid version {}".format(version.strip()))
        weights = []
        e = 0
        for line in f:
            line = line.decode('ascii').strip()
            if not line:
                continue
            e += 1
            weight = list(map(float, line.split(' ')))
            weights.append(weight)
            if e == 2:
                filters = len(line.split(' '))
                print("Channels", filters)
        blocks = e - (4 + 14)
        if blocks % 8 != 0:
            raise ValueError("Inconsistent number of weights in the file - e = {}".format(e))
        blocks //= 8
        print("Blocks", blocks)
    return (filters, blocks, weights)


def _parse_layer(layer):
    weights = np.frombuffer(layer.params, dtype=np.uint16).astype(np.float32) / np.iinfo(np.uint16).max
    return weights * (layer.max_val - layer.min_val) + layer.min_val

def _parse_conv_block(block):
    return list(map(_parse_layer, (block.weights, block.biases, block.bn_means, block.bn_stddivs)))

def _parse_residual_block(block):
    return _parse_conv_block(block.conv1) + _parse_conv_block(block.conv2)

def _parse_net_weights(weights):
    num_blocks = len(weights.residual)

    input_block = _parse_conv_block(weights.input)

    # number of filters is given by size of bias
    num_filters = input_block[1].shape[0]

    residual_blocks = [layer for block in map(_parse_residual_block, weights.residual) for layer in block]

    policy_conv = _parse_conv_block(weights.policy)
    policy_fc_weights = _parse_layer(weights.ip_pol_w)
    policy_fc_bias = _parse_layer(weights.ip_pol_b)

    policy_weights = policy_conv + [policy_fc_weights, policy_fc_bias]

    value_conv = _parse_conv_block(weights.value)
    value_fc1_weights = _parse_layer(weights.ip1_val_w)
    value_fc1_bias = _parse_layer(weights.ip1_val_b)
    value_fc2_weights = _parse_layer(weights.ip2_val_w)
    value_fc2_bias = _parse_layer(weights.ip2_val_b)

    value_weights = value_conv + [value_fc1_weights, value_fc1_bias, value_fc2_weights, value_fc2_bias]

    return (num_filters, num_blocks, input_block + residual_blocks + policy_weights + value_weights)


def read_weights_file_pb2(opener):
    import numpy as np
    import net_pb2

    with opener('rb') as f:
        net = net_pb2.Net()
        net.ParseFromString(f.read())

    if net.format.weights_encoding != net_pb2.Format.LINEAR16:
        raise Exception("Unknown encoding of weights file for weights.")

    return _parse_net_weights(net.weights)
    