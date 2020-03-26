from prob_unet.unet_blocks import *
import torch.nn.functional as F

class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, initializers, apply_last_layer=True, padding=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = apply_last_layer
        self.contracting_path = nn.ModuleList()

        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]

            if i == 0:
                pool = False
            else:
                pool = True

            self.contracting_path.append(DownConvBlock(input, output, initializers, padding, pool=pool))

        self.upsampling_path = nn.ModuleList()

        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, initializers, padding))

        if self.apply_last_layer:
            self.last_layer = nn.Conv3d(output, num_classes, kernel_size=1)
            #nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            #nn.init.normal_(self.last_layer.bias)


    def forward(self, x, val):
        blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                blocks.append(x)

        for i, up in enumerate(self.upsampling_path):
            x = up(x, blocks[-i-1])

        del blocks

        #Used for saving the activations and plotting
        if val:
            self.activation_maps.append(x)
        
        if self.apply_last_layer:
            x =  self.last_layer(x)

        return x


class NoNewNet(nn.Module):
    def __init__(self, in_channels, out_channels, final_sigmoid=False, f_maps=64, layer_order='crg', num_groups=8,
                 **kwargs):
        super(NoNewNet, self).__init__()
        channels = out_channels
        self.levels = 5

        # create encoder levels
        encoderModules = []
        encoderModules.append(EncoderModule(in_channels, channels, channels, False, True))
        for i in range(self.levels - 2):
            encoderModules.append(EncoderModule(channels * pow(2, i), channels * pow(2, i+1), channels, True, True))
        encoderModules.append(EncoderModule(channels * pow(2, self.levels - 2), channels * pow(2, self.levels - 1), channels, True, False))
        self.encoders = nn.ModuleList(encoderModules)

        # create decoder levels
        decoderModules = []
        decoderModules.append(DecoderModule(channels * pow(2, self.levels - 1), channels * pow(2, self.levels - 2), channels, True, False))
        for i in range(self.levels - 2):
            decoderModules.append(DecoderModule(channels * pow(2, self.levels - i - 2), channels * pow(2, self.levels - i - 3), channels, True, True))
        decoderModules.append(DecoderModule(channels, channels, channels, False, True))
        self.decoders = nn.ModuleList(decoderModules)

    def forward(self, x):
        # x, y_out = self.se(x)
        inputStack = []
        for i in range(self.levels):
            x = self.encoders[i](x)
            if i < self.levels - 1:
                inputStack.append(x)

        for i in range(self.levels):
            x = self.decoders[i](x)
            if i < self.levels - 1:
                x = x + inputStack.pop()
        return x


class EncoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, channels=30, maxpool=False, secondConv=True, hasDropout=False):
        super(EncoderModule, self).__init__()
        groups = min(outChannels, channels)
        self.maxpool = maxpool
        self.secondConv = secondConv
        self.hasDropout = hasDropout
        self.conv1 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, outChannels)
        # self.bn = nn.BatchNorm3d(outChannels)
        # self.in1 = nn.InstanceNorm3d(outChannels)
        # self.se = SELayer(outChannels)
        if secondConv:
            self.conv2 = nn.Conv3d(outChannels, outChannels, 3, padding=1, bias=False)
            self.gn2 = nn.GroupNorm(groups, outChannels)
            # self.bn2 = nn.BatchNorm3d(outChannels)
            # self.in2 = nn.InstanceNorm3d(outChannels)
        if hasDropout:
            self.dropout = nn.Dropout3d(0.2, True)

    def forward(self, x):
        if self.maxpool:
            x = F.max_pool3d(x, 2)
        doInplace = True and not self.hasDropout
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.leaky_relu(x, inplace=doInplace)
        if self.hasDropout:
            x = self.dropout(x)
        if self.secondConv:
            x = F.leaky_relu(self.gn1(self.conv2(x)), inplace=doInplace)
        # x = self.se(x)
        return x


class DecoderModule(nn.Module):
    def __init__(self, inChannels, outChannels, channels=30, upsample=False, firstConv=True):
        super(DecoderModule, self).__init__()
        groups = min(outChannels, channels)
        self.upsample = upsample
        self.firstConv = firstConv
        if firstConv:
            self.conv1 = nn.Conv3d(inChannels, inChannels, 3, padding=1, bias=False)
            self.gn1 = nn.GroupNorm(groups, inChannels)
        self.conv2 = nn.Conv3d(inChannels, outChannels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, outChannels)

    def forward(self, x):
        if self.firstConv:
            x = F.leaky_relu(self.gn1(self.conv1(x)), inplace=True)
        x = F.leaky_relu(self.gn2(self.conv2(x)), inplace=True)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        return x
