import math
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.ops import roi_pool
from models.tools import add_ml_decoder_head, FastAvgPool2d

model_urls = {
    "mobilenet_v2": "https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1",
}


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.avgpool =  nn.AdaptiveAvgPool2d((1,1))  # FastAvgPool2d(flatten=True)
        self.to_ft = FastAvgPool2d(flatten=True)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, n_class)
        self.dec = nn.Linear(64,20)

        self._initialize_weights()

    def forward(self, x, **kwargs):
        oo = []
        rli = []
        for i in range(len(self.features)): 
            x = self.features[i](x)
            if i == len(self.features) // 2:
                kr = self.avgpool(x).flatten(1)
                uskdo = self.dec(kr)
            oo.append(x)
        m = x
        x = self.avgpool(x)  # 11111111111
        if "loc" in kwargs.keys():
            B = x.shape[0] 
            for idx in range(kwargs["iter"]): 
                lo =  kwargs["loc"][idx*B:idx*B+B]   # (B 4)   
                roip1 = roi_pool(x, lo, output_size=(1,1), spatial_scale=1/32)  # 2222
                rli.append(roip1)
            rli = torch.cat(rli,dim=0)
            rli = rli.view(rli.shape[0], -1)  # 333333
            sli = rli
            rli = self.classifier(rli) # (3*B C) 
             
        x = x.view(x.shape[0], -1)   # 4444444444
        h = x
        x = self.classifier(x) 
        
        if kwargs['usew'] == 'sd':
            if "loc" in kwargs.keys(): 
                return [x, rli, sli, None, None] 
            else:
                return [x, h, None] 
        elif kwargs['usew'] == 'byot':
            return x, [o1, o2, o3, o4], [map1, map2, map3, map4]
        elif kwargs['usew'] == 'nnn':
            return [x]
        elif kwargs["usew"] == 'uskd':
            return x, uskdo
        elif kwargs["usew"] == 'ddgsd':
            return x, r
        elif kwargs['usew'] in ['dlb','pskd']:
            return [x]
        elif kwargs['usew'] == "ud":
            return x,m
        else:
            return [x]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(num_classes, pretrained=True):
    model = MobileNetV2(width_mult=1,n_class=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["mobilenet_v2"])
        state_dict.pop("classifier.weight")
        state_dict.pop("classifier.bias")
        model.load_state_dict(state_dict,strict=False)
    # add_ml_decoder_head(model, num_classes=num_classes, num_of_groups=num_classes, decoder_embedding=768)  # 888888888888888888
    return model
