import torch
import torch.nn as nn

# config
# tuple = (out_channels, kernel, stride)
# list ["BLOCK", num_repeats]
#   "R": Residual
# blocks are repeated implementations of the same layers
# string: scale prediction / upsampling

default_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["R", 1],
    (128, 3, 2),
    ["R", 2],
    (256, 3, 2),
    ["RS", 8],
    (512, 3, 2),
    ["RS", 8],
    (1024, 3, 2),
    ["R", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class YoloConfigError(TypeError):
    pass


class CNN1DBlock(nn.Module):
    """A basic CNN block

    in_channels: number of input features per pixel (i.e. 3 for RGB)
    out_channels: number of output_features
    batch_norm_activation: applies leaky relu and batch normalization, else linear
    **kwargs passed to torch.nn.Conv1d. Notable kwargs:
        + stride: convolutional stride
        + kernel_size: convolutional kernel size
        + padding: padding added to input
        + padding_mode: how to pad
    """
    def __init__(self, in_channels, out_channels, batch_norm_activation=True,
                 *args, **kwargs) -> None:
        super().__init__(*args)
        # print("padding", kwargs.get("padding"))
        self.conv = nn.Conv1d(in_channels, out_channels,
                              bias=not batch_norm_activation,
                              **kwargs)
        # print(self.conv)
        self.bn = nn.BatchNorm1d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_activation = batch_norm_activation

    def forward(self, x):
        if self.use_bn_activation:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):
    """"""
    def __init__(self, channels, use_residual=True, num_repeats=1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNN1DBlock(channels, channels//2,
                               kernel_size=1, padding=0),
                    CNN1DBlock(channels//2, channels,
                               kernel_size=3, padding=1)
                )
            ]
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes,
                 num_anchors_per_scale=1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_anchors_per_scale = num_anchors_per_scale
        # objectness, anchor_x, width
        self.prediction_features_1d = 3
        self.pred = nn.Sequential(
            CNN1DBlock(in_channels, 2*in_channels,
                       kernel_size=3, padding=1),
            CNN1DBlock(2*in_channels,
                       (self.num_anchors_per_scale
                        *(num_classes
                          + self.prediction_features_1d)
                        ),
                       batch_norm_activation=False,
                       kernel_size=1)
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (self.pred(x)
                # 0 Number of examples in batch
                # 1 num anchors
                # 2 num clases + n bbox predictors
                # 3 scale x
                .reshape(x.shape[0],
                         self.num_anchors_per_scale,
                         (self.num_classes
                          + self.prediction_features_1d),
                         x.shape[2])
                # permute it such that classes is at the end
                .permute(0, 1, 3, 2))
        # output: (N_exp, N_anch, scale,
        #   (objectness, anch_x, width, *classes))

class Yolo1DV3(nn.Module):
    """Put everything together
    in_channels (1 for our data, 3 in yolo (RGB data))
    num_classes (number of detectable classes)
    """
    def __init__(self, in_channels=1, num_classes=4,
                 num_anchors_per_scale=1,
                 config=default_config,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors_per_scale = num_anchors_per_scale
        self.config = config
        self.layers, self.route_connection_locations = self._create_conv_layers()


    def forward(self, x):
        """forward step through network

        Args:
            x (torch.Tensor): network input

        Returns:
            list[torch.Tensor]: list of scale predictions
        """
        outputs = []
        route_connections = []

        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, ScalePrediction):
                # make a predition and continue to main branch
                # any code below `continue` will not run!
                outputs.append(layer(x))
                continue

            # step one layer
            x = layer(x)

            # handle route connections
            if layer_idx in self.route_connection_locations:
                # create route connection
                route_connections.append(x)
            elif isinstance(layer, nn.Upsample):
                # concatenate with last route connection
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        route_connection_locations = []
        in_channels = self.in_channels
        for module_idx, module in enumerate(self.config):
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(CNN1DBlock(
                    in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if kernel_size == 3 else 0,
                ))
                in_channels = out_channels
            elif isinstance(module, list):
                block_type, num_repeats = module
                if not isinstance(block_type, str):
                    raise YoloConfigError("Repeatable blocks should be lists with a string in the"
                                          "0-th index and an integer in 1-st index"
                                          "(list[str, int])")
                if "R" in block_type:
                    layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
                if "S" in block_type:
                    # add a skip connection
                    route_connection_locations.append(module_idx)
            elif isinstance(module, str):
                if module == "S":
                    # detection layer
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNN1DBlock(in_channels, in_channels//2,
                                   kernel_size=1,
                                   ),
                        ScalePrediction(in_channels//2,
                                        num_classes=self.num_classes,
                                        num_anchors_per_scale=self.num_anchors_per_scale)
                    ]
                    in_channels = in_channels // 2
                elif module == "U":
                    # upsample layer
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3
        return layers, route_connection_locations

if __name__ == "__main__":
    # quick test
    num_classes = 20
    data_length = 416
    num_anchors_per_scale = 1
    model = Yolo1DV3(num_classes=num_classes,
                     in_channels=1,
                     num_anchors_per_scale=1)
    # batchsize, channels, data_length
    x = torch.randn((2, 1, data_length))
    print(x.dtype)
    out = model(x)
    assert out[0].shape == (2, num_anchors_per_scale, data_length//32, num_classes + 3), "error in small scale"
    assert out[1].shape == (2, num_anchors_per_scale, data_length//16, num_classes + 3), "error in medium scale"
    assert out[2].shape == (2, num_anchors_per_scale, data_length//8, num_classes + 3), "error in large scale"
    print("Successfully built Yolo1DV3!")

