import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask():
    '''
    Class that holds the mask properties

    hard: the hard/binary mask (1 or 0), 4-dim tensor
    soft (optional): the float mask, same shape as hard
    active_positions: the amount of positions where hard == 1
    total_positions: the total amount of positions 
                        (typically batch_size * output_width * output_height)
    '''

    def __init__(self, hard, soft=None):
        assert hard.dim() == 4
        assert hard.shape[1] == 1
        assert soft is None or soft.shape == hard.shape

        self.hard = hard
        self.active_positions = torch.sum(hard)  # this must be kept backpropagatable!
        self.total_positions = hard.numel()
        self.soft = soft

        self.flops_per_position = 0

    def size(self):
        return self.hard.shape

    def __repr__(self):
        return f'Mask with {self.active_positions}/{self.total_positions} positions, and {self.flops_per_position} accumulated FLOPS per position'


class MaskUnit(nn.Module):
    ''' 
    Generates the mask and applies the gumbel softmax trick 
    '''

    def __init__(self, channels, stride=1, dilate_stride=1):
        super(MaskUnit, self).__init__()
        self.maskconv = Squeeze(in_channels=channels, stride=stride)
        # self.gumbel = Gumbel()
        self.expandmask = ExpandMask(stride=dilate_stride)

    def forward(self, x):
        soft = self.maskconv(x)
        return soft


## Gumbel

# class Gumbel(nn.Module):
#     '''
#     Returns differentiable discrete outputs. Applies a Gumbel-Softmax trick on every element of x.
#     '''
#
#     def __init__(self, eps=1e-8):
#         super(Gumbel, self).__init__()
#         self.eps = eps
#
#     def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
#         if not self.training:  # no Gumbel noise during inference
#             return (x >= 0).float()
#
#         if gumbel_noise:
#             eps = self.eps
#             U1, U2 = torch.rand_like(x), torch.rand_like(x)
#             g1, g2 = -torch.log(-torch.log(U1 + eps) + eps), - \
#                 torch.log(-torch.log(U2 + eps) + eps)
#             x = x + g1 - g2
#
#         soft = torch.sigmoid(x / gumbel_temp)
#         hard = ((soft >= 0.5).float() - soft).detach() + soft
#         assert not torch.any(torch.isnan(hard))
#         return hard


## Mask convs
class Squeeze(nn.Module):
    """ 
    Squeeze module to predict masks 
    """

    def __init__(self, in_channels, stride=1):
        super(Squeeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1, bias=True)
        self.conv = nn.Conv2d(in_channels, 1, stride=stride,
                              kernel_size=3, padding=1, bias=True)

    # input: B C H W
    # ret: B 1 H W + B 1
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, 1, 1, 1)
        z = self.conv(x)
        return z + y.expand_as(z)


class ExpandMask(nn.Module):
    def __init__(self, stride, padding=1):
        super(ExpandMask, self).__init__()
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        assert x.shape[1] == 1

        if self.stride > 1:
            self.pad_kernel = torch.zeros((1, 1, self.stride, self.stride), device=x.device)
            self.pad_kernel[0, 0, 0, 0] = 1
        self.dilate_kernel = torch.ones((1, 1, 1 + 2 * self.padding, 1 + 2 * self.padding), device=x.device)

        x = x.float()
        if self.stride > 1:
            x = F.conv_transpose2d(x, self.pad_kernel, stride=self.stride, groups=x.size(1))
        x = F.conv2d(x, self.dilate_kernel, padding=self.padding, stride=1)
        return x > 0.5
