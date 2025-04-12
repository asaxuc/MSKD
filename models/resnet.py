import sys
import os
import torchvision.models as models 
from torch.hub import load_state_dict_from_url
import torch
import random
import torch.nn.functional as F
from models.maskconv import MaskUnit 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from typing import Type, Callable, Union, List, Optional
from axu import Istr_1, Istr_2, Istr_3, Istr_4
import torch.nn as nn
from torch import Tensor
from stn import STM
from models.ml_decoder import MLDecoder
from torchvision.ops import roi_pool
from models.tools import add_ml_decoder_head, FastAvgPool2d

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self, 
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 =  nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 =  norm_layer(self.inplanes)
        self.relu =  nn.ReLU()
        self.maxpool =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 =  self._make_layer(block, 64, layers[0])
        self.layer2 =  self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 =  self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 =  self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_ft = FastAvgPool2d(flatten=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes) 
        # self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        # self.features = nn.Sequential(
        #     model.conv1,
        #     model.bn1,
        #     model.relu,
        #     model.maxpool,
        #     model.layer1,
        #     model.layer2,
        #     model.layer3,
        #     model.layer4,
        # )

        # self.features = nn.Sequential()
        # self.features.add_module('conv1', model.conv1)
        # self.features.add_module('bn1', model.bn1)
        # self.features.add_module('relu', model.relu)
        # self.features.add_module('maxpool', model.maxpool)
        # self.features.add_module('layer1', model.layer1)
        # self.features.add_module('layer2', model.layer2)
        # self.features.add_module('layer3', model.layer3)
        # self.features.add_module('layer4', model.layer4)
 

        # self.dec1 = Istr_1(block, num_classes)  # MLDecoder(num_classes=num_classes, initial_num_features=64, num_of_groups=num_classes, decoder_embedding=256)
        # self.dec2 = Istr_2(block, num_classes)  # MLDecoder(num_classes=num_classes, initial_num_features=128, num_of_groups=num_classes, decoder_embedding=256)
        # self.dec3 = Istr_3(block, num_classes)  # MLDecoder(num_classes=num_classes, initial_num_features=256, num_of_groups=num_classes, decoder_embedding=256)
        # self.dec4 = Istr_4(block, num_classes)  # MLDecoder(num_classes=num_classes, initial_num_features=512, num_of_groups=num_classes, decoder_embedding=256)

        # self.dec1 = MLDecoder(self.uniemb, num_classes=num_classes, initial_num_features=64, num_of_groups=num_classes, decoder_embedding=256)
        # self.dec2 = MLDecoder(self.uniemb, num_classes=num_classes, initial_num_features=128, num_of_groups=num_classes, decoder_embedding=256)
        # self.dec3 = MLDecoder(self.uniemb, num_classes=num_classes, initial_num_features=256, num_of_groups=num_classes, decoder_embedding=256)
        # self.dec4 = MLDecoder(self.uniemb, num_classes=num_classes, initial_num_features=512, num_of_groups=num_classes, decoder_embedding=256)

        # self.dec1 = nn.Linear(64, num_classes)
        # self.d1_2 = MaskUnit(64)
        # self.c1_2 = nn.Conv2d(64, 128, 1, 1, 0)

        self.dec2 = nn.Linear(128, num_classes)
        # self.d2_3 = MaskUnit(128)
        # self.c2_3 = nn.Conv2d(128, 256, 1, 1, 0)

        # self.dec3 = nn.Linear(256, num_classes)
        # self.d3_4 = MaskUnit(256)
        # self.c3_4 = nn.Conv2d(256, 512, 1, 1, 0)

        # self.dec4 = nn.Linear(512, num_classes)
 

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type] 
        

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers) 
 
    def _forward_impl(self, k: Tensor, **kwargs):
        # See note [TorchScript super()] 
        rli = [] 
        oo = []
        x = k
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
 
        x = self.layer1(x)  
        # if kwargs["usew"] == 'ddgsd':
        #     oo.append(x)
        # o1 = o1.view(o1.shape[0], -1)
        # o1 = self.fc1(o1)
 
        x = self.layer2(x)  
        if kwargs["usew"] == 'uskd':
            k = nn.AdaptiveAvgPool2d((1, 1))(x).flatten(1)
            uskdo = self.dec2(k)
        # if kwargs["usew"] == 'ddgsd':
        #     oo.append(x)
        # o2 = o2.view(o2.shape[0], -1)
        # o2 = self.fc2(o2)
         
        x = self.layer3(x) 
        # if kwargs["usew"] == 'ddgsd':
        #     oo.append(x) 
        # o3 = o3.view(o3.shape[0], -1)
        # o3 = self.fc3(o3) 
        
        x = self.layer4(x)   
        # if kwargs["usew"] == 'ddgsd':
        #     oo.append(x)
        # o4 = o4.view(o4.shape[0], -1)
        # o4 = self.fc4(o4) 
        m = x
        x = self.avgpool(m)  # 1111111
        if "loc" in kwargs.keys():
            B = k.shape[0] 
            for idx in range(kwargs["iter"]): 
                lo =  kwargs["loc"][idx*B:idx*B+B]   # (B 4)   
                roip1 = roi_pool(m, lo, output_size=(1,1), spatial_scale=1/32)  # 2222
                rli.append(roip1)
            rli = torch.cat(rli,dim=0)
            rli = rli.view(rli.shape[0], -1)  # 333333
            ssl = rli
            rli  = self.fc(rli) # (3*B C)  #  7777777777777777
             
                
        x = x.view(x.shape[0], -1) #  4444444
        r = x
        # la = self.fc1((map1+map2+map3+map4)/4)
        # o1, o2, o3, o4 = [None] * 4
        # map1, map2, map3, map4 = [None] * 4
        x = self.fc(x)   #  555555555555
           #  666666666666 
        if kwargs['usew'] == 'sd':
            if "loc" in kwargs.keys(): 
                return [x, rli, ssl, None, None] 
            else:
                return [x, r, None] 
        elif kwargs['usew'] == 'byot':
            return x, [o1, o2, o3, o4], [map1, map2, map3, map4]
        elif kwargs['usew'] == 'uskd':
            return x, uskdo
        elif kwargs['usew'] == "ddgsd":
            return x, r
        elif kwargs['usew'] == 'nnn':
            return [x]
        elif kwargs['usew'] in ['dlb','pskd']:
            return [x]
        elif kwargs['usew'] == "ud":
            return x, m
        else:
            return [x]

    def forward(self, x: Tensor,  **kwargs):
        return self._forward_impl(x, **kwargs)
    
    def clear(self):
        self.la = []


def resnet18(num_classes, pretrained=False):
    net = ResNet(block=BasicBlock, num_classes=num_classes, layers=[2, 2, 2, 2])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet18"])
        net.load_state_dict(state_dict)
    add_ml_decoder_head(net, num_classes=num_classes, num_of_groups=num_classes, decoder_embedding=768)
    return net
 
def resnet34(num_classes, pretrained=True):
    net = ResNet(block=BasicBlock, num_classes=num_classes, layers=[3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet34"])
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        net.load_state_dict(state_dict,strict=False)
     
    # add_ml_decoder_head(net, num_classes=num_classes, num_of_groups=num_classes, decoder_embedding=768)  # 88888888888888888
    return net


def resnet50(num_classes, pretrained=False):
    net = ResNet(block=Bottleneck, num_classes=num_classes, layers=[3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["resnet50"])
        net.load_state_dict(state_dict)
    add_ml_decoder_head(net, num_classes=num_classes, num_of_groups=num_classes, decoder_embedding=768)
    return net


def resnet101(num_classes, pretrained=False):
    model34 = models.resnet34(pretrained=pretrained)
    net = ResNet(block=Bottleneck, num_classes=num_classes, layers=[3, 4, 23, 3])
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls["resnet101"])
    #     net.load_state_dict(state_dict)
    add_ml_decoder_head(net, num_classes=num_classes, num_of_groups=num_classes, decoder_embedding=768)
    return net


def wrn50(num_classes, pretrained=False):
    net = ResNet(block=Bottleneck, num_classes=num_classes, layers=[3, 4, 6, 3], width_per_group=64 * 2)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["wide_resnet50_2"])
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        net.load_state_dict(state_dict,strict=False)
    # add_ml_decoder_head(net, num_classes=num_classes, num_of_groups=num_classes, decoder_embedding=768)
    return net


def wrn101(num_classes, pretrained=False):
    net = ResNet(block=Bottleneck, num_classes=num_classes, layers=[3, 4, 23, 3], width_per_group=64 * 2)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls["wide_resnet101_2"])
    #     net.load_state_dict(state_dict)
    add_ml_decoder_head(net, num_classes=num_classes, num_of_groups=num_classes, decoder_embedding=768)
    return net

def get_resnext50_32x4d(num_classes, pretrained=False):
    net = ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, num_classes=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["wide_resnet50_2"])
        state_dict.pop("fc.weight")
        state_dict.pop("fc.bias")
        net.load_state_dict(state_dict,strict=False)
    # add_ml_decoder_head(net, num_classes=num_classes, num_of_groups=num_classes, decoder_embedding=768)
    return net