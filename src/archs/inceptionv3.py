import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Tuple, Optional


__all__ = ['InceptionV3', 'inceptionv3']

pretrained_settings = {
    'inceptionv3': {
        'imagenet': {
            'url': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
            'input_space': 'RGB',
            'input_size': 299,
            'input_range': [0, 1],
            'input_mean': [0.5, 0.5, 0.5],
            'input_std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
    },
}


class InceptionV3(nn.Module):
    def __init__(self, 
                 num_classes: int = 1000, 
                 aux_logits: bool = True,
                 dropout: float = 0.5,
                 init_weights: Optional[bool] = None,
    ) -> None:
        super(InceptionV3, self).__init__()
        self.aux_logits = aux_logits

        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=0)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1, stride=1, padding=0)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        
        self.AuxLogits: Optional[nn.Module] = None
        if self.training and self.aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # x:  [n, 3, 299, 299]
        x = self.Conv2d_1a_3x3(x) # --> [n, 32, 149, 149]
        x = self.Conv2d_2a_3x3(x) # --> [n, 32, 147, 147]
        x = self.Conv2d_2b_3x3(x) # --> [n, 64, 147, 147]
        x = self.maxpool1(x) # --> [n, 64, 73, 73]
        x = self.Conv2d_3b_1x1(x) # --> [n, 80, 73, 73]
        x = self.Conv2d_4a_3x3(x) # --> [n, 192, 71, 71]
        x = self.maxpool2(x) # --> [n, 192, 35, 35]
        x = self.Mixed_5b(x) # --> [n, 256, 35, 35]
        x = self.Mixed_5c(x) # --> [n, 288, 35, 35]
        x = self.Mixed_5d(x) # --> [n, 288, 35, 35]
        x = self.Mixed_6a(x) # --> [n, 768, 17, 17]
        x = self.Mixed_6b(x) # --> [n, 768, 17, 17]
        x = self.Mixed_6c(x) # --> [n, 768, 17, 17]
        x = self.Mixed_6d(x) # --> [n, 768, 17, 17]
        x = self.Mixed_6e(x) # --> [n, 768, 17, 17]

        if self.training and self.aux_logits:
            self._out_aux = self.AuxLogits(x)

        x = self.Mixed_7a(x) # --> [n, 1280, 8, 8]
        x = self.Mixed_7b(x) # --> [n, 2048, 8, 8]
        x = self.Mixed_7c(x) # --> [n, 2048, 8, 8]
        
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)) # --> [n, 2048, 1, 1]
        x = self.dropout(x) # --> [n, 2048, 1, 1]
        x = torch.flatten(x, 1) # --> [n, 2048]
        x = self.fc(x)  # --> [n, 1000]

        # if self.training and self.aux_logits:
        #     aux = self._out_aux
        #     self._out_aux = None
        #     return x, aux

        return x


class InceptionA(nn.Module):
    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], dim=1)
        return outputs
        

class InceptionB(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = torch.cat([branch3x3, branch3x3dbl, branch_pool], dim=1)
        return outputs


class InceptionC(nn.Module):
    def __init__(self, in_channels: int, channels_7x7: int) -> None:
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7_1 = BasicConv2d(in_channels, channels_7x7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(channels_7x7, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_1 = BasicConv2d(in_channels, channels_7x7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(channels_7x7, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], dim=1)
        return outputs


class InceptionD(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)
        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = torch.cat([branch3x3, branch7x7x3, branch_pool], 1)
        return outputs
    

class InceptionE(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], dim=1)
        return outputs


class InceptionAux(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x: Tensor) -> Tensor:
        # x: [n, 768, 17, 17]
        x = F.avg_pool2d(x, kernel_size=5, stride=3) # --> [n, 768, 5, 5]
        x = self.conv0(x) # [n, 128, 5, 5]
        x = self.conv1(x) # [n, 768, 1, 1]
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)) # [n, 768, 1, 1]
        x = torch.flatten(x, 1) # [n, 768]
        x = self.fc(x) # [n, 1000]
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


def inceptionv3(pretrained='imagenet'):
    """
    InceptionV3 model architecture.
    """
    if pretrained is not None:
        print('=> Loading from pretrained model: {}'.format(pretrained))
        settings = pretrained_settings['inceptionv3'][pretrained]

        model = InceptionV3()
        state_dict = model_zoo.load_url(settings['url'])
        for k, v in state_dict.items():
            state_dict[k] = torch.squeeze(v, dim=0)
        model.load_state_dict(state_dict)
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.input_mean = settings['input_mean']
        model.input_std = settings['input_std']
    else:
        raise NotImplementedError
    return model


if __name__ == '__main__':
    model = inceptionv3()