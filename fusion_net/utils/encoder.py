import torch
import fusion_net.utils.net_utils as net_utils


'''
Encoders fusion_net/utils/encoder.py
'''

class FusionNetEncoder(torch.nn.Module):
    '''
    FusionNet encoder with skip connections
    Arg(s):
        n_layer : int
            number of layer for encoder
        input_channels_image : int
            number of channels in input data
        input_channels_depth : int
            number of channels in input data
        n_filters_per_block : list[int]
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
        fusion_type : str
            add, weight
    '''

    def __init__(self,
                 n_layer=18,
                 input_channels_image=3,
                 input_channels_depth=3,
                 n_filters_encoder_image=[32, 64, 128, 256, 256],
                 n_filters_encoder_depth=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 fusion_type='add'):
        super(FusionNetEncoder, self).__init__()

        self.fusion_type = fusion_type

        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
        else:
            raise ValueError('Only supports 18, 34 layer architecture')

        resnet_block = net_utils.ResNetBlock

        assert len(n_filters_encoder_image) == len(n_filters_encoder_depth)

        for n in range(len(n_filters_encoder_image) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]

        network_depth = len(n_filters_encoder_image)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert network_depth == len(n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        activation_func = net_utils.activation_func(activation_func)

        # Resolution 1/1 -> 1/2
        self.conv1_image = net_utils.Conv2d(
            input_channels_image,
            n_filters_encoder_image[filter_idx],
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv1_depth = net_utils.Conv2d(
            input_channels_depth,
            n_filters_encoder_depth[filter_idx],
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if fusion_type == 'add':
            self.conv1_project = net_utils.Conv2d(
                n_filters_encoder_depth[filter_idx],
                n_filters_encoder_image[filter_idx],
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight':

            self.conv1_weight = net_utils.Conv2d(
                n_filters_encoder_depth[filter_idx],
                n_filters_encoder_depth[filter_idx],
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight_and_project':

            self.conv1_weight = net_utils.Conv2d(
                n_filters_encoder_depth[filter_idx],
                n_filters_encoder_image[filter_idx],
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv1_project = net_utils.Conv2d(
                n_filters_encoder_depth[filter_idx],
                n_filters_encoder_image[filter_idx],
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]

        self.blocks2_image, self.blocks2_depth = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if fusion_type == 'add':
            self.conv2_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight':

            self.conv2_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight_and_project':

            self.conv2_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv2_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]

        self.blocks3_image, self.blocks3_depth = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if fusion_type == 'add':
            self.conv3_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight':

            self.conv3_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight_and_project':

            self.conv3_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv3_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]

        self.blocks4_image, self.blocks4_depth = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if fusion_type == 'add':
            self.conv4_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight':

            self.conv4_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight_and_project':

            self.conv4_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv4_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        in_channels_image, out_channels_image = [
            n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
        ]

        in_channels_depth, out_channels_depth = [
            n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
        ]

        self.blocks5_image, self.blocks5_depth = self._make_layer(
            network_block=resnet_block,
            n_block=n_blocks[block_idx],
            in_channels_image=in_channels_image,
            in_channels_depth=in_channels_depth,
            out_channels_image=out_channels_image,
            out_channels_depth=out_channels_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        if fusion_type == 'add':
            self.conv5_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        elif fusion_type == 'weight':

            self.conv5_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_depth,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

        if fusion_type == 'weight_and_project':

            self.conv5_weight = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('sigmoid'),
                use_batch_norm=use_batch_norm)

            self.conv5_project = net_utils.Conv2d(
                out_channels_depth,
                out_channels_image,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=net_utils.activation_func('linear'),
                use_batch_norm=use_batch_norm)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters_encoder_image):

            in_channels_image, out_channels_image = [
                n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
            ]

            in_channels_depth, out_channels_depth = [
                n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
            ]

            self.blocks6_image, self.blocks6_depth = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                out_channels_image=out_channels_image,
                out_channels_depth=out_channels_depth,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            if fusion_type == 'add':
                self.conv6_project = net_utils.Conv2d(
                    out_channels_depth,
                    out_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('linear'),
                    use_batch_norm=use_batch_norm)

            if fusion_type == 'weight_and_project':

                self.conv6_weight = net_utils.Conv2d(
                    out_channels_depth,
                    out_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('sigmoid'),
                    use_batch_norm=use_batch_norm)

                self.conv6_project = net_utils.Conv2d(
                    out_channels_depth,
                    out_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('linear'),
                    use_batch_norm=use_batch_norm)
        else:
            self.blocks6_image = None
            self.blocks6_depth = None
            self.conv6_weight = None
            self.conv6_project = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters_encoder_image):

            in_channels_image, out_channels_image = [
                n_filters_encoder_image[filter_idx-1], n_filters_encoder_image[filter_idx]
            ]

            in_channels_depth, out_channels_depth = [
                n_filters_encoder_depth[filter_idx-1], n_filters_encoder_depth[filter_idx]
            ]

            self.blocks7_image, self.blocks7_depth = self._make_layer(
                network_block=resnet_block,
                n_block=n_blocks[block_idx],
                in_channels_image=in_channels_image,
                in_channels_depth=in_channels_depth,
                out_channels_image=out_channels_image,
                out_channels_depth=out_channels_depth,
                stride=2,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            if fusion_type == 'weight_and_project':

                self.conv7_weight = net_utils.Conv2d(
                    out_channels_depth,
                    out_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('sigmoid'),
                    use_batch_norm=use_batch_norm)

                self.conv7_project = net_utils.Conv2d(
                    out_channels_depth,
                    out_channels_image,
                    kernel_size=1,
                    stride=1,
                    weight_initializer=weight_initializer,
                    activation_func=net_utils.activation_func('linear'),
                    use_batch_norm=use_batch_norm)
        else:
            self.blocks7_image = None
            self.blocks7_depth = None
            self.conv7_weight = None
            self.conv7_project = None

    def _make_layer(self,
                    network_block,
                    n_block,
                    in_channels_image,
                    in_channels_depth,
                    out_channels_image,
                    out_channels_depth,
                    stride,
                    weight_initializer,
                    activation_func,
                    use_batch_norm):
        '''
        Creates a layer
        Arg(s):
            network_block : Object
                block type
            n_block : int
                number of blocks to use in layer
            in_channels_image : int
                number of channels in image branch
            in_channels_depth : int
                number of channels in depth branch
            out_channels_image : int
                number of output channels in image branch
            out_channels_depth : int
                number of output channels in depth branch
            stride : int
                stride of convolution
            weight_initializer : str
                kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
            activation_func : func
                activation function after convolution
            use_batch_norm : bool
                if set, then applied batch normalization
        '''

        blocks_image = []
        blocks_depth = []

        for n in range(n_block):

            if n == 0:
                stride = stride
            else:
                in_channels_image = out_channels_image
                in_channels_depth = out_channels_depth
                stride = 1

            block_image = network_block(
                in_channels=in_channels_image,
                out_channels=out_channels_image,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks_image.append(block_image)

            block_depth = network_block(
                in_channels=in_channels_depth,
                out_channels=out_channels_depth,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks_depth.append(block_depth)

        blocks_image = torch.nn.Sequential(*blocks_image)
        blocks_depth = torch.nn.Sequential(*blocks_depth)

        return blocks_image, blocks_depth

    def forward(self, image, depth):
        '''
        Forward input x through the ResNet model
        Arg(s):
            image : torch.Tensor
            depth : torch.Tensor
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''

        layers = []

        # Resolution 1/1 -> 1/2
        conv1_image = self.conv1_image(image)
        conv1_depth = self.conv1_depth(depth)

        if self.fusion_type == 'add':
            conv1_project = self.conv1_project(conv1_depth)
            conv1 = conv1_project + conv1_image
        elif self.fusion_type == 'weight':
            conv1_weight = self.conv1_weight(conv1_depth)
            conv1 = conv1_weight * conv1_depth + conv1_image
        elif self.fusion_type == 'weight_and_project':
            conv1_weight = self.conv1_weight(conv1_depth)
            conv1_project = self.conv1_project(conv1_depth)
            conv1 = conv1_weight * conv1_project + conv1_image
        elif self.fusion_type == 'concat':
            conv1 = torch.cat([conv1_depth, conv1_image], dim=1)
        else:
            raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

        layers.append(conv1)

        # Resolution 1/2 -> 1/4
        max_pool_image = self.max_pool(conv1_image)
        max_pool_depth = self.max_pool(conv1_depth)

        blocks2_image = self.blocks2_image(max_pool_image)
        blocks2_depth = self.blocks2_depth(max_pool_depth)

        if self.fusion_type == 'add':
            conv2_project = self.conv2_project(blocks2_depth)
            blocks2 = conv2_project + blocks2_image
        elif self.fusion_type == 'weight':
            conv2_weight = self.conv2_weight(blocks2_depth)
            blocks2 = conv2_weight * blocks2_depth + blocks2_image
        elif self.fusion_type == 'weight_and_project':
            conv2_weight = self.conv2_weight(blocks2_depth)
            conv2_project = self.conv2_project(blocks2_depth)
            blocks2 = conv2_weight * conv2_project + blocks2_image
        elif self.fusion_type == 'concat':
            blocks2 = torch.cat([blocks2_image, blocks2_depth], dim=1)
        else:
            raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

        layers.append(blocks2)

        # Resolution 1/4 -> 1/8
        blocks3_image = self.blocks3_image(blocks2_image)
        blocks3_depth = self.blocks3_depth(blocks2_depth)

        if self.fusion_type == 'add':
            conv3_project = self.conv3_project(blocks3_depth)
            blocks3 = conv3_project + blocks3_image
        elif self.fusion_type == 'weight':
            conv3_weight = self.conv3_weight(blocks3_depth)
            blocks3 = conv3_weight * blocks3_depth + blocks3_image
        elif self.fusion_type == 'weight_and_project':
            conv3_weight = self.conv3_weight(blocks3_depth)
            conv3_project = self.conv3_project(blocks3_depth)
            blocks3 = conv3_weight * conv3_project + blocks3_image
        elif self.fusion_type == 'concat':
            blocks3 = torch.cat([blocks3_image, blocks3_depth], dim=1)
        else:
            raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

        layers.append(blocks3)

        # Resolution 1/8 -> 1/16
        blocks4_image = self.blocks4_image(blocks3_image)
        blocks4_depth = self.blocks4_depth(blocks3_depth)

        if self.fusion_type == 'add':
            conv4_project = self.conv4_project(blocks4_depth)
            blocks4 = conv4_project + blocks4_image
        elif self.fusion_type == 'weight':
            conv4_weight = self.conv4_weight(blocks4_depth)
            blocks4 = conv4_weight * blocks4_depth + blocks4_image
        elif self.fusion_type == 'weight_and_project':
            conv4_weight = self.conv4_weight(blocks4_depth)
            conv4_project = self.conv4_project(blocks4_depth)
            blocks4 = conv4_weight * conv4_project + blocks4_image
        elif self.fusion_type == 'concat':
            blocks4 = torch.cat([blocks4_image, blocks4_depth], dim=1)
        else:
            raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

        layers.append(blocks4)

        # Resolution 1/16 -> 1/32
        blocks5_image = self.blocks5_image(blocks4_image)
        blocks5_depth = self.blocks5_depth(blocks4_depth)

        if self.fusion_type == 'add':
            conv5_project = self.conv5_project(blocks5_depth)
            blocks5 = conv5_project + blocks5_image
        elif self.fusion_type == 'weight':
            conv5_weight = self.conv5_weight(blocks5_depth)
            blocks5 = conv5_weight * blocks5_depth + blocks5_image
        elif self.fusion_type == 'weight_and_project':
            conv5_weight = self.conv5_weight(blocks5_depth)
            conv5_project = self.conv5_project(blocks5_depth)
            blocks5 = conv5_weight * conv5_project + blocks5_image
        elif self.fusion_type == 'concat':
            blocks5 = torch.cat([blocks5_image, blocks5_depth], dim=1)
        else:
            raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

        layers.append(blocks5)

        # Resolution 1/32 -> 1/64
        if self.blocks6_image is not None and self.blocks6_depth is not None:
            blocks6_image = self.blocks6_image(blocks5_image)
            blocks6_depth = self.blocks6_depth(blocks5_depth)

            if self.fusion_type == 'add':
                conv6_project = self.conv6_project(blocks6_depth)
                blocks6 = conv6_project + blocks6_image
            elif self.fusion_type == 'weight':
                conv6_weight = self.conv6_weight(blocks6_depth)
                blocks6 = conv6_weight * blocks6_depth + blocks6_image
            elif self.fusion_type == 'weight_and_project':
                conv6_weight = self.conv6_weight(blocks6_depth)
                conv6_project = self.conv6_project(blocks6_depth)
                blocks6 = conv6_weight * conv6_project + blocks6_image
            elif self.fusion_type == 'concat':
                blocks6 = torch.cat([blocks6_image, blocks6_depth], dim=1)
            else:
                raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

            layers.append(blocks6)

        # Resolution 1/64 -> 1/128
        if self.blocks7_image is not None and self.blocks7_depth is not None:
            blocks7_image = self.blocks7_image(blocks6_image)
            blocks7_depth = self.blocks7_depth(blocks6_depth)

            if self.fusion_type == 'add':
                conv7_project = self.conv7_project(blocks7_depth)
                blocks7 = conv7_project + blocks7_image
            elif self.fusion_type == 'weight':
                conv7_weight = self.conv7_weight(blocks7_depth)
                blocks7 = conv7_weight * blocks7_depth + blocks7_image
            elif self.fusion_type == 'weight_and_project':
                conv7_weight = self.conv7_weight(blocks7_depth)
                conv7_project = self.conv7_project(blocks7_depth)
                blocks7 = conv7_weight * conv7_project + blocks7_image
            elif self.fusion_type == 'concat':
                blocks7 = torch.cat([blocks7_image, blocks7_depth], dim=1)
            else:
                raise ValueError('Unsupported fusion type: {}'.format(self.fusion_type))

            layers.append(blocks7)

        return layers[-1], layers[:-1]

