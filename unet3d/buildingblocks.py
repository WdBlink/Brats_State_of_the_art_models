import torch
from torch import nn as nn
from torch.nn import functional as F
from unet3d.config import load_config
from torch.nn import init
from .sync_batchnorm import SynchronizedBatchNorm3d

# initalize the module
def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def conv3d(in_channels, out_channels, kernel_size, bias, padding=1, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride)


def fcn(in_dim, out_dim):
    return nn.Linear(in_dim, out_dim)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
            # number of groups must be less or equal the number of channels
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crg', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))


class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    This block can be used instead of standard DoubleConv in the Encoder module.
    Motivated by: https://arxiv.org/pdf/1706.00120.pdf

    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()

        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order,
                                num_groups=num_groups)

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='crg',
                 num_groups=8):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='crg', num_groups=8):
        super(Decoder, self).__init__()
        if basic_module == DoubleConv:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining
            # works correctly
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            # use nearest neighbor interpolation and concatenation joining
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
            # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
            x = torch.cat((encoder_features, x), dim=1)
        else:
            # use ConvTranspose3d and summation joining
            x = self.upsample(x)
            x += encoder_features

        x = self.basic_module(x)
        return x


class FinalConv(nn.Sequential):
    """
    A module consisting of a convolution layer (e.g. Conv3d+ReLU+GroupNorm3d) and the final 1x1 convolution
    which reduces the number of channels to 'out_channels'.
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8):
        super(FinalConv, self).__init__()

        # conv1
        self.add_module('SingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups))

        # in the last layer a 1×1 convolution reduces the number of output channels to out_channels
        final_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.add_module('final_conv', final_conv)


class GreenBlock(nn.Module):
    """
    green_block(inp, filters, name=None)
    ------------------------------------
    Implementation of the special residual block used in the paper. The block
    consists of two (GroupNorm --> ReLu --> 3x3x3 non-strided Convolution)
    units, with a residual connection from the input `inp` to the output. Used
    internally in the model. Can be used independently as well.

    Parameters
    ----------
    `inp`: An keras.layers.layer instance, required
        The keras layer just preceding the green block.
    `filters`: integer, required
        No. of filters to use in the 3D convolutional block. The output
        layer of this green block will have this many no. of channels.
    `data_format`: string, optional
        The format of the input data. Must be either 'chanels_first' or
        'channels_last'. Defaults to `channels_first`, as used in the paper.
    `name`: string, optional
        The name to be given to this green block. Defaults to None, in which
        case, keras uses generated names for the involved layers. If a string
        is provided, the names of individual layers are generated by attaching
        a relevant prefix from [GroupNorm_, Res_, Conv3D_, Relu_, ], followed
        by _1 or _2.

    Returns
    -------
    `out`: A keras.layers.Layer instance
        The output of the green block. Has no. of channels equal to `filters`.
        The size of the rest of the dimensions remains same as in `inp`.
    """

    def __init__(self, input_channels, output_channels):
        super(GreenBlock, self).__init__()
        self.conv3d_1 = conv3d(input_channels, output_channels, padding=0, kernel_size=1, bias=True)
        self.gn1 = nn.GroupNorm(input_channels, input_channels)
        self.act_1 = nn.ReLU()

        self.conv3d_2 = conv3d(input_channels, output_channels, kernel_size=3, bias=True)
        self.gn2 = nn.GroupNorm(output_channels, output_channels)
        self.act_2 = nn.ReLU()

        self.conv3d_3 = conv3d(output_channels, output_channels, kernel_size=3, bias=True)

    def forward(self, x):
        inp_res = self.conv3d_1(x)
        x = self.gn1(x)
        x = self.act_1(x)

        x = self.conv3d_2(x)
        x = self.gn2(x)
        x = self.act_2(x)

        x = self.conv3d_3(x)
        out = inp_res + x
        return out


class DownBlock(nn.Module):
    """
    A module down sample the feature map
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, input_channels, output_channels, order="cbr", num_groups=8):
        super(DownBlock, self).__init__()
        self.convblock1 = SingleConv(input_channels, output_channels, order=order, num_groups=num_groups)
        self.convblock2 = SingleConv(output_channels, output_channels, order=order, num_groups=num_groups)
        # self.downsample = conv3d(output_channels, output_channels, kernel_size=3, bias=True, stride=2)
        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        conv1 = self.convblock1(x)
        conv2 = self.convblock2(conv1)
        down = self.downsample(conv2)
        return down, conv2


class UpBlock(nn.Module):
    """
        A module down sample the feature map
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of the convolving kernel
            order (string): determines the order of layers, e.g.
                'cr' -> conv + ReLU
                'crg' -> conv + ReLU + groupnorm
            num_groups (int): number of groups for the GroupNorm
        """

    def __init__(self, input_channels, output_channels):
        super(UpBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = conv3d(input_channels, output_channels, kernel_size=1, bias=True, padding=0)

    def forward(self, x):
        _, c, w, h, d = x.size()
        upsample1 = F.upsample(x, [2 * w, 2 * h, 2 * d], mode='trilinear')
        upsample = self.conv1(upsample1)
        return upsample


class unetConv3(nn.Module):

    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv3, self).__init__()
        self.n = n

        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, stride, padding),
                                     nn.BatchNorm3d(out_size),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv3d(in_size, out_size, ks, stride, padding),
                                     nn.ReLU(inplace=True))
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        for m in self.children():
            init_weights(m, 'kaiming')

    def forward(self, input):
        x = input
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class unetUp(nn.Module):

    def __init__(self, in_size, out_size, is_deconv=False, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv3(in_size + (n_concat - 2) * out_size, out_size, is_batchnorm=False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(in_size, out_size, kernel_size=1)
            )

        for m in self.children():
            if m.__class__.__name__.find('unetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


# class EncoderModule(nn.Module):
#     def __init__(self, inChannels, outChannels, maxpool=False, secondConv=True, hasDropout=False):
#         super(EncoderModule, self).__init__()
#         groups = min(outChannels, 30)
#         self.maxpool = maxpool
#         self.secondConv = secondConv
#         self.hasDropout = hasDropout
#         self.conv1 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
#         self.gn1 = nn.GroupNorm(groups, outChannels)
#         if secondConv:
#             self.conv2 = nn.Conv3d(outChannels, outChannels, 3, padding=1, bias=False)
#             self.gn2 = nn.GroupNorm(groups, outChannels)
#         if hasDropout:
#             self.dropout = nn.Dropout3d(0.2, True)
#
#         # for m in self.children():
#         #     init_weights(m, init_type='kaiming')
#
#     def forward(self, x):
#         if self.maxpool:
#             x = F.max_pool3d(x, 2)
#         doInplace = True and not self.hasDropout
#         x = F.relu(self.gn1(self.conv1(x)), inplace=doInplace)
#         if self.hasDropout:
#             x = self.dropout(x)
#         if self.secondConv:
#             x = F.relu(self.gn2(self.conv2(x)), inplace=True)
#         return x

class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, maxpool=False, secondConv=True, hasDropout=False):
        super(EncoderModule, self).__init__()
        groups = min(outChannels, 30)
        self.maxpool = maxpool
        self.secondConv = secondConv
        self.hasDropout = hasDropout
        self.conv1 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        # self.gn1 = nn.GroupNorm(groups, outChannels)
        self.bn1 = SynchronizedBatchNorm3d(outChannels)
        # self.in1 = nn.InstanceNorm3d(outChannels)
        if secondConv:
            self.conv2 = nn.Conv3d(outChannels, outChannels, 3, padding=1, bias=False)
            # self.gn2 = nn.GroupNorm(groups, outChannels)
            self.bn2 = SynchronizedBatchNorm3d(outChannels)
            # self.in2 = nn.InstanceNorm3d(outChannels)
        if hasDropout:
            self.dropout = nn.Dropout3d(0.2, True)

    def forward(self, x):
        if self.maxpool:
            x = F.max_pool3d(x, 2)
        doInplace = True and not self.hasDropout
        # x = F.leaky_relu(self.gn1(self.conv1(x)), inplace=doInplace)
        # x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=doInplace)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        if self.hasDropout:
            x = self.dropout(x)
        if self.secondConv:
            # x = F.leaky_relu(self.gn2(self.conv2(x)), inplace=doInplace)
            # x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=doInplace)
            x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=doInplace)
        return x

# class DecoderModule(nn.Module):
#     def __init__(self, inChannels, outChannels, upsample=False, firstConv=True):
#         super(DecoderModule, self).__init__()
#         groups = min(outChannels, 30)
#         self.upsample = upsample
#         self.firstConv = firstConv
#         if firstConv:
#             self.conv1 = nn.Conv3d(inChannels, inChannels, 3, padding=1, bias=False)
#             self.gn1 = nn.GroupNorm(groups, inChannels)
#         self.conv2 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
#         self.gn2 = nn.GroupNorm(groups, outChannels)
#
#         # for m in self.children():
#         #     init_weights(m, init_type='kaiming')
#
#     def forward(self, x):
#         if self.firstConv:
#             x = F.relu(self.gn1(self.conv1(x)), inplace=True)
#         x = F.relu(self.gn2(self.conv2(x)), inplace=True)
#         if self.upsample:
#             x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
#         return x


class DecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, upsample=False, firstConv=True):
        super(DecoderModule, self).__init__()
        groups = min(outChannels, 30)
        self.upsample = upsample
        self.firstConv = firstConv
        if firstConv:
            self.conv1 = nn.Conv3d(inChannels, inChannels, 3, padding=1, bias=False)
            # self.gn1 = nn.GroupNorm(groups, inChannels)
            self.bn1 = SynchronizedBatchNorm3d(inChannels)
            # self.in1 = nn.InstanceNorm3d(inChannels)
        self.conv2 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        # self.gn2 = nn.GroupNorm(groups, outChannels)
        self.bn2 = SynchronizedBatchNorm3d(outChannels)
        # self.in2 = nn.InstanceNorm3d(outChannels)

    def forward(self, x):
        if self.firstConv:
            # x = F.leaky_relu(self.gn1(self.conv1(x)), inplace=True)
            # x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
            x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        # x = F.leaky_relu(self.gn2(self.conv2(x)), inplace=True)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x


class VaeBlock(nn.Module):
    """
        A module that carry out vae regularization
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of the convolving kernel
            order (string): determines the order of layers, e.g.
                'cr' -> conv + ReLU
                'crg' -> conv + ReLU + groupnorm
            num_groups (int): number of groups for the GroupNorm
        """

    def __init__(self, input_channels, output_channels, order="cbr", num_groups=8):
        super(VaeBlock, self).__init__()
        self.conv_block = SingleConv(input_channels, 1, order=order, num_groups=num_groups)
        self.fcn = nn.Linear(512, 128)
        self.fcn1 = nn.Linear(128, 64)
        self.fcn2 = nn.Linear(128, 64)
        self.fcn3 = nn.Linear(128, 4096)
        self.conv1 = conv3d(1, 128, kernel_size=1, bias=True, padding=0)
        # self.greenblock = GreenBlock(128, 128)
        self.conv2 = conv3d(128, 64, kernel_size=1, bias=True, padding=0)
        # self.greenblock1 = GreenBlock(64, 64)
        self.conv3 = conv3d(64, 32, kernel_size=1, bias=True, padding=0)
        # self.greenblock2 = GreenBlock(32, 32)
        self.conv4 = conv3d(32, output_channels, kernel_size=1, bias=True, padding=0)

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x)
        x = self.fcn(x)
        z_mean = self.fcn1(x)
        z_var = self.fcn2(x)
        x = self.sampling([z_mean, z_var])
        x = torch.reshape(x, (-1, 128))

        x = self.fcn3(x)
        x = torch.reshape(x, (-1, 1, 16, 16, 16))
        x = self.conv1(x)
        x = F.upsample(x, size=[2 * x.size(2), 2 * x.size(3), 2 * x.size(4)], mode='trilinear')
        # x = self.greenblock(x)

        x = self.conv2(x)
        x = F.upsample(x, size=[2 * x.size(2), 2 * x.size(3), 2 * x.size(4)], mode='trilinear')
        # x = self.greenblock1(x)

        x = self.conv3(x)
        x = F.upsample(x, size=[2 * x.size(2), 2 * x.size(3), 2 * x.size(4)], mode='trilinear')
        # x = self.greenblock2(x)

        x = self.conv4(x)

        return x, z_mean, z_var

    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments
            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """
        config = load_config()
        z_mean, z_var = args
        batch = 2
        dim = z_mean.size(0)
        epsilon = torch.randn([batch, dim]).to(config['device'])
        return z_mean + torch.exp(0.5 * z_var) * epsilon


class CaeBlock(nn.Module):
    """
        A module that carry out CAE regularization
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of the convolving kernel
            order (string): determines the order of layers, e.g.
                'cr' -> conv + ReLU
                'crg' -> conv + ReLU + groupnorm
            num_groups (int): number of groups for the GroupNorm
        """

    def __init__(self, input_channels, output_channels, order="cbr", num_groups=8):
        super(CaeBlock, self).__init__()
        self.conv_block = SingleConv(input_channels, 1, order=order, num_groups=num_groups)
        self.conv1 = conv3d(1, 128, kernel_size=1, bias=True, padding=0)
        # self.greenblock = GreenBlock(128, 128)
        self.conv2 = conv3d(128, 64, kernel_size=1, bias=True, padding=0)
        # self.greenblock1 = GreenBlock(64, 64)
        self.conv3 = conv3d(64, 32, kernel_size=1, bias=True, padding=0)
        # self.greenblock2 = GreenBlock(32, 32)
        self.conv4 = conv3d(32, 16, kernel_size=1, bias=True, padding=0)
        # self.greenblock3 = GreenBlock(16, 16)
        self.conv5 = conv3d(16, output_channels, kernel_size=1, bias=True, padding=0)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.conv1(x)
        x = F.upsample(x, size=[2 * x.size(2), 2 * x.size(3), 2 * x.size(4)], mode='trilinear')
        # x = self.greenblock(x)

        x = self.conv2(x)
        x = F.upsample(x, size=[2 * x.size(2), 2 * x.size(3), 2 * x.size(4)], mode='trilinear')
        # x = self.greenblock1(x)

        x = self.conv3(x)
        x = F.upsample(x, size=[2 * x.size(2), 2 * x.size(3), 2 * x.size(4)], mode='trilinear')
        # x = self.greenblock2(x)

        x = self.conv4(x)
        x = F.upsample(x, size=[2 * x.size(2), 2 * x.size(3), 2 * x.size(4)], mode='trilinear')
        # x = self.greenblock3(x)

        x = self.conv5(x)
        x = torch.sigmoid(x)
        x = x * 2 - 1

        return x


class ResEncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, maxpool=False, secondConv=True, hasDropout=False):
        super(ResEncoderModule, self).__init__()
        groups = min(outChannels, 30)
        self.maxpool = maxpool
        self.secondConv = secondConv
        self.hasDropout = hasDropout

        self.greenblock1 = GreenBlock(inChannels, outChannels)
        self.conv1 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)

        self.gn1 = nn.GroupNorm(groups, outChannels)
        if secondConv:
            self.conv2 = nn.Conv3d(outChannels, outChannels, 3, padding=1, bias=False)
            self.greenblock2 = GreenBlock(outChannels, outChannels)
            self.gn2 = nn.GroupNorm(groups, outChannels)
        if hasDropout:
            self.dropout = nn.Dropout3d(0.2, True)

        self.convpool = nn.Conv3d(inChannels, inChannels, kernel_size=3, stride=2, bias=False, padding=1)

    def forward(self, x):
        if self.maxpool:
            # x = F.max_pool3d(x, 2)
            x = self.convpool(x)
        doInplace = True and not self.hasDropout
        x = self.greenblock1(x)
        if self.hasDropout:
            x = self.dropout(x)
        if self.secondConv:
            x = self.greenblock2(x)
        return x


class ResDecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, upsample=False, firstConv=True):
        super(ResDecoderModule, self).__init__()
        groups = min(outChannels, 30)
        self.upsample = upsample
        self.firstConv = firstConv
        if firstConv:
            self.conv1 = nn.Conv3d(inChannels, inChannels, 3, padding=1, bias=False)
            self.greenblock1 = GreenBlock(inChannels, inChannels)
            self.gn1 = nn.GroupNorm(groups, inChannels)
        self.conv2 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        self.greenblock2 = GreenBlock(inChannels, outChannels)
        self.gn2 = nn.GroupNorm(groups, outChannels)

    def forward(self, x):
        if self.firstConv:
            x = self.greenblock1(x)
        x = self.greenblock2(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3, 3, 1), padding_size=(1, 1, 0),
                 init_stride=(1, 1, 1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True), )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True), )
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True), )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3, 3, 3), padding_size=(1, 1, 1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)


class GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class MedicaNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(MedicaNetBasicBlock, self).__init__()
        self.conv1 = self.conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def conv3x3x3(self, in_planes, out_planes, stride=1, dilation=1):
        # 3x3x3 convolution with padding
        return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=3,
            dilation=dilation,
            stride=stride,
            padding=dilation,
            bias=False)


class MedicaNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(MedicaNetBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=3):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=3):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, low_in_channels, high_in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_type=None,psp_size=(1,3,6,8)):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = low_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = high_in_channels
        self.pool = nn.MaxPool3d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm3d(self.key_channels),
            nn.ReLU()
        )
        self.f_query = nn.Sequential(
            nn.Conv3d(in_channels=high_in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            SynchronizedBatchNorm3d(self.key_channels),
            nn.ReLU()
        )
        self.f_value = nn.Conv3d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv3d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, low_feats, high_feats):
        batch_size, h, w = high_feats.size(0), high_feats.size(2), high_feats.size(3)
        # if self.scale > 1:
        #     x = self.pool(x)

        value = self.psp(self.f_value(low_feats))

        query = self.f_query(high_feats).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(low_feats)
        # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.psp(key)  # .view(batch_size, self.key_channels, -1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *high_feats.size()[2:])
        context = self.W(context)
        return context


class SelfAttentionBlock3D(_SelfAttentionBlock):
    def __init__(self, low_in_channels, high_in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_type=None,psp_size=(1,3,6,8)):
        super(SelfAttentionBlock3D, self).__init__(low_in_channels,
                                                   high_in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale,
                                                   norm_type,
                                                   psp_size=psp_size
                                                   )


class APNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), norm_type=None,psp_size=(1,3,6,8)):
        super(APNB, self).__init__()
        self.stages = []
        self.norm_type = norm_type
        self.psp_size=psp_size
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            SynchronizedBatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock3D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    self.norm_type,
                                    self.psp_size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class AFNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout,
                 sizes=([1]), norm_type=None,psp_size=(1,3,6,8)):
        super(AFNB, self).__init__()
        self.stages = []
        self.norm_type = norm_type
        self.psp_size=psp_size
        self.stages = nn.ModuleList(
            [self._make_stage([low_in_channels, high_in_channels], out_channels, key_channels, value_channels, size) for
             size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(out_channels + high_in_channels, out_channels, kernel_size=1, padding=0),
            SynchronizedBatchNorm3d(out_channels),
            nn.Dropout3d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock3D(in_channels[0],
                                    in_channels[1],
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    self.norm_type,
                                    psp_size=self.psp_size)

    def forward(self, low_feats, high_feats):
        priors = [stage(low_feats, high_feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, high_feats], 1))
        return output


class asymmetric_non_local_network(nn.Sequential):
    def __init__(self, configer):
        super(asymmetric_non_local_network, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = Hello
        # low_in_channels, high_in_channels, out_channels, key_channels, value_channels, dropout
        self.fusion = AFNB(1024, 2048, 2048, 256, 256, dropout=0.05, sizes=([1]),
                           norm_type=self.configer.get('network', 'norm_type'))
        # extra added layers
        self.context = nn.Sequential(
            nn.Conv3d(2048, 512, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm3d(512),
            nn.ReLU(),
            APNB(in_channels=512, out_channels=512, key_channels=256, value_channels=256,
                 dropout=0.05, sizes=([1]), norm_type=self.configer.get('network', 'norm_type'))
        )
        self.cls = nn.Conv3d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1),
            SynchronizedBatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv3d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn(x[-2])
        x = self.fusion(x[-2], x[-1])
        x = self.context(x)
        x = self.cls(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3), x_.size(4)), mode="trilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3), x_.size(4)), mode="trilinear", align_corners=True)
        return aux_x, x