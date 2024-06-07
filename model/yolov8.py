"""1D implementation of Yolov8

This follows the example of Ultralitics Yolov8
"""
import copy
import math
import os
import warnings

import torch

class DFL(torch.nn.Module):
    """Distributional Focal Loss

    https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, in_channels=16):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(in_channels, dtype=torch.float)
        self.conv.weight.data[:] = torch.nn.Parameter(x.view(1, in_channels, 1, 1))
        self.in_channels = in_channels

    def forward(self, x):
        """Applies transformer layer on input and returns a tensor"""
        batch, chan, anch = x.shape
        return self.conv(x.view(batch, 4, self.in_channels, anch)
                           .transpose(2, 1)
                           .softmax(1).view(batch, 4, anch))


class Conv(torch.nn.Module):
    """Conv1D Module

    Basic convolution module with batch normalization and SiLU activation

    Batch normalization normalizes the inputs by re-centering and re-scaling

    The SiLU activation function or Sigmoid Linear Unit is a cousin of
    the RELU and LeakyRELU activation fuctions. It is defined as
    SiLU(x) = sig(x) * x
    note that for x >> 0 sig(x) = 1 and for x << 0 sig(x) = 0
    therefore SiLU(x) ≈ RELU for |x| >> 0
    however, it does leak some gradent for x < 0, like LeakyRELU
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel=1, stride=1, pad=0,
                 activation=torch.nn.SiLU(),
                 ):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel,
                                    stride=stride,
                                    padding=pad)
        self.batch_norm = torch.nn.BatchNorm1d(out_channels)
        if not isinstance(activation, torch.nn.Module):
            warnings.warn("Invalid activation supplied, set activation to Identity")
            activation = torch.nn.Identity()
        self.activation = activation

    def forward(self, x):
        return self.SiLU(self.batch_norm(self.conv(x)))

class SPP(torch.nn.Module):
    """Spatial Pyramid Pooling

    Spatial Pyramid Pooling removes the fixed size input constraint of a network.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel=5):
        super().__init__()
        intermediate_channels = in_channels//2
        self.conv_1 = Conv(in_channels, intermediate_channels, kernel=1, stride=1, pad=0)
        self.conv_2 = Conv(intermediate_channels*4, out_channels, kernel=1, stride=1, pad=0)
        self.mp = torch.nn.MaxPool1d(kernel_size=kernel,
                                     stride=1,
                                     padding=kernel//2)

    def forward(self, x):
        x_1_conv = self.conv_1(x)
        x_2_mp = self.mp(x_1_conv)
        x_3_mp = self.mp(x_2_mp)
        x_4_mp = self.mp(x_3_mp)
        x_5_concat = torch.concat((x_2_mp, x_3_mp, x_4_mp))
        return self.conv_2(x_5_concat)


class Bottleneck(torch.nn.Module):
    """Bottleneck Module

    The bottleneck transforms the feature space into some hidden state and back into
    the original state.

    Shortcuts (skip connections) can be added. These are implemented by
    addition (not concatenation).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shortcut: bool=False,
                 p_hidden: float=0.5,
                 ):
        super().__init__()
        intermediate_channels = int(out_channels*p_hidden)
        self.conv_1 = Conv(in_channels, intermediate_channels, kernel=3, stride=1, pad=0)
        self.conv_2 = Conv(intermediate_channels, out_channels, kernel=3, stride=1, pad=0)
        self.shortcut = shortcut
        self.__can_add = in_channels == out_channels

    def forward(self, x):
        out = self.conv_1(self.conv_2(x))
        if self.shortcut and self.__can_add:
            out += x
        return out


class C2f(torch.nn.Module):
    """Fast C2 Module

    The C2 Module is a CSP Bottleneck with 2 convolutions

    A CSP Bottleneck is a "Cros Stage Parital Network" type bottleneck.

    A CSP Network addresses duplicate gradient information problem
    allowing for great reduction in network complexity while mainaining
    model accuracy. Duplicate gradient information can occur due to
    concatenation of many similarly processed layers (I think?).
    """

    def __init__(self, in_channels, out_channels, n_repeats, shortcut):
        super().__init__()
        conv2_in_channels = int((n_repeats+2) * out_channels//2)
        self.conv_1 = Conv(in_channels=in_channels, out_channels=out_channels,
                             kernel=1, stride=1, pad=0)
        self.conv_2 = Conv(in_channels=conv2_in_channels, out_channels=out_channels,
                             kernel=1, stride=1, pad=0)
        self.bnecks = torch.nn.ModuleList((
            Bottleneck(in_channels=int(out_channels//2),
                                  out_channels=out_channels,
                                  shortcut=shortcut,
                                  p_hidden=1.0)
            ) for _ in range(n_repeats))
        self.n_repeats = n_repeats

    def forward(self, x):
        concats = list(self.conv_1(x).chunk(2, 1))
        concats.extend(bneck(concats[-1]) for bneck in self.bnecks)
        return self.conv_2(torch.concat(concats, 1))


class AFDetect(torch.nn.Module):
    """Anchor Free Detector"""

    def __init__(self, num_classes, channels=tuple()):
        super().__init__()
        self.num_classes = num_classes
        self.num_detection_layers = len(channels)
        self.reg_max = 16  # DFL channels (?)
        self.number_of_output_per_anchor = (self.num_classes
                                            + self.reg_max * 4)
        # strides are computed during build
        self.stride = torch.zeros(self.num_detection_layers)
        self.channels_2 = max((16, channels[0]//4, self.reg_max//4))
        self.channels_3 = max((channels[0], min(self.num_classes, 100)))
        self.cv2 = torch.nn.ModuleList(
            torch.nn.Sequential(Conv(ch, self.channels_2, kernel=3),
                                Conv(self.channels_2, self.channels_2, kernel=3),
                                Conv(self.channels_2, 4 * self.reg_max, kernel=1),
                                ) for ch in channels
        )
        self.cv3 = torch.nn.ModuleList(
            torch.nn.Sequential(Conv(ch, self.channels_3, kernel=3),
                                Conv(self.channels_3, self.channels_3, kernel=3),
                                Conv(self.channels_3, self.num_classes, kernel=1),
                                ) for ch in channels
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else torch.nn.Identity()

    def forward(self, x):
        input_shape = x[0].shape
        for layer_idx in range(self.num_detection_layers):
            x[layer_idx].concat((self.cv2[layer_idx](x[layer_idx]),
                                 self.cv3[layer_idx](x[layer_idx])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != input_shape:
            self.anchors, self.strides = (x.transpose(0, 1)
                                          for x in make_anchors(x, self.stride, 0.5)
                                          # TODO make_anchors ?
                                          )

        x_cat = torch.cat([xi.view(shape[0], self.number_of_output_per_anchor)])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for a, b, s in zip(self.cv2, self.cv3, self.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:self.num_classes] = \
                math.log(5 / self.num_classes / (640 / s) ** 2)


def load_config(path: os.PathLike) -> dict:
    # TODO load config from yaml
    pass

def parse_model(config: dict, input_channels: int, verbose: bool=False):
    # TODO parse model
    pass

class Yolo1Dv8(torch.nn.Module):
    """1D implementation of a Yolov8-like network"""

    def __init__(self,
                 config: dict | os.PathLike=None,
                 input_channels: int=1,
                 num_classes: None | int=None,
                 verbose: bool=False,
                 ):
        self.config = config if isinstance(config, dict) else load_config(config)
        conf_nc = config.get("nc")
        if (num_classes is None
            and num_classes != conf_nc):
            self.config["nc"] = num_classes
            warnings.warn("Overriding config nc="
                          f"{conf_nc} with nc="
                          f"{num_classes}. (nc = number of classes)")
        self.model, self.save = parse_model(copy.deepcopy(self.config),
                                            input_channels=input_channels,
                                            verbose=verbose)
        self.names = {i: f'{i}' for i in range(self.config["nc"])}
        self.inplace = self.config.get("inplace", True)

        # build strides
        model_head = self.model[-1]  # ?
        if isinstance(model_head, AFDetect):
            min_stride = 256
            model_head.inplace = self.inplace
            forward = lambda x: self.forward(x)
            model_head.stride = torch.tensor(
                # TODO does this generalize to 1D?
                [min_stride / x.shape[-2]
                 for x in forward(torch.zeros(1, input_channels,
                                              min_stride, min_stride)
                                  )
                 ]
                )
            self.stride = model_head.stride
            model_head.bias_init()
        else:
            self.stride = torch.Tensor([32])
        # TODO
        self.initalize_weights()

    def initialize_weights():
        raise NotImplementedError("initialize_weights has not yet been implemented!")

    def _predict_augment(self, x):
        raise NotImplementedError("_predict_augment has not yet been implemented!")

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        else:
            return self.predict(x, *args, **kwargs)

    def predict(self, x, augment=False):
        if augment:
            return self._predict_augment()
        return self._predict_once(x)

    def _predict_once(self, x):
        """Forward pass"""
        for layer in self.model:
            if layer.f != -1:
                # TODO idk whats supposed to be going on here...
                # probably concatenation or something like that
                # x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                pass
            x = layer(x)
        return x
