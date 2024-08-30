from torch import nn
from ..models.gsf import GSF
from ..models.gsm import GSM
from ..models.tsm import TSM
LAYER_BUILDER_DICT=dict()


def parse_expr(expr):
    parts = expr.split('<=')
    return parts[0].split(','), parts[1], parts[2].split(',')


def get_basic_layer(info, channels=None, conv_bias=False):
    id = info['id']
    attr = info['attrs'] if 'attrs' in info else list()

    out, op, in_vars = parse_expr(info['expr'])
    assert(len(out) == 1)
    assert(len(in_vars) == 1)
    mod, out_channel, = LAYER_BUILDER_DICT[op](attr, channels, conv_bias)

    return id, out[0], mod, out_channel, in_vars[0]


def build_conv(attr, channels=None, conv_bias=False):
    out_channels = attr['num_output']
    ks = attr['kernel_size'] if 'kernel_size' in attr else (attr['kernel_h'], attr['kernel_w'])
    if 'pad' in attr or 'pad_w' in attr and 'pad_h' in attr:
        padding = attr['pad'] if 'pad' in attr else (attr['pad_h'], attr['pad_w'])
    else:
        padding = 0
    if 'stride' in attr or 'stride_w' in attr and 'stride_h' in attr:
        stride = attr['stride'] if 'stride' in attr else (attr['stride_h'], attr['stride_w'])
    else:
        stride = 1

    conv = nn.Conv2d(channels, out_channels, ks, stride, padding, bias=conv_bias)

    return conv, out_channels


def build_pooling(attr, channels=None, conv_bias=False):
    method = attr['mode']
    pad = attr['pad'] if 'pad' in attr else 0
    try:
        global_pool = attr['global']
    except:
        global_pool = False
    if method == 'max':
        pool = nn.MaxPool2d(attr['kernel_size'], attr['stride'], pad,
                            ceil_mode=True) # all Caffe pooling use ceil model
    elif method == 'ave':
        if global_pool:
            pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            pool = nn.AvgPool2d(attr['kernel_size'], attr['stride'], pad,
                                ceil_mode=True)  # all Caffe pooling use ceil model

    else:
        raise ValueError("Unknown pooling method: {}".format(method))

    return pool, channels

def build_identity(attr, channels=None, conv_bias=False):

    return nn.Identity(), channels


def build_relu(attr, channels=None, conv_bias=False):
    return nn.ReLU(inplace=True), channels


def build_bn(attr, channels=None, conv_bias=False):
    return nn.BatchNorm2d(channels, momentum=0.1), channels


def build_linear(attr, channels=None, conv_bias=False):
    return nn.Linear(channels, attr['num_output']), channels

def build_dropout(attr, channels=None, conv_bias=False):
    return nn.Dropout(p=attr['dropout_ratio']), channels

def build_gsm(info, channels=None, conv_bias=False, num_segments=3):
    id = info['id']
    attr = info['attrs'] if 'attrs' in info else list()
    out, op, in_vars = parse_expr(info['expr'])
    out_channels = attr['fPlane']
    gsm = GSM(out_channels, num_segments=num_segments)
    return id, out[0], gsm, out_channels, in_vars[0]

def build_gsf(info, num_segments=3, gsf_ch_ratio=100):
    id = info['id']
    attr = info['attrs'] if 'attrs' in info else list()
    out, op, in_vars = parse_expr(info['expr'])
    out_channels = attr['fPlane']
    gsf = GSF(out_channels, num_segments=num_segments, gsf_ch_ratio=gsf_ch_ratio)
    return id, out[0], gsf, out_channels, in_vars[0]

def build_tsm(info, num_segments=3, n_div=8):
    id = info['id']
    attr = info['attrs'] if 'attrs' in info else list()
    out, op, in_vars = parse_expr(info['expr'])
    out_channels = attr['fPlane']
    gsf = TSM(num_segments=num_segments, n_div=n_div)
    return id, out[0], gsf, out_channels, in_vars[0]

LAYER_BUILDER_DICT['Convolution'] = build_conv

LAYER_BUILDER_DICT['Pooling'] = build_pooling

LAYER_BUILDER_DICT['ReLU'] = build_relu

LAYER_BUILDER_DICT['Dropout'] = build_dropout

LAYER_BUILDER_DICT['BN'] = build_bn

LAYER_BUILDER_DICT['InnerProduct'] = build_linear

LAYER_BUILDER_DICT['Identity'] = build_identity
