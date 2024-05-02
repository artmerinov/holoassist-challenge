import torch
from torch import nn
import timm 
import torch.nn.functional as F

class ViTBackbone(nn.Module):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        if modality == 'flow':
            self.pre = nn.Conv2d(2, 3, 1)
        elif modality == 'flow-s5':
            self.pre = nn.Conv2d(10, 3, 1)
        self.model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True)
    
    def forward(self, x):
        features = []
        for i in range(x.size(1)):
            if 'flow' in self.modality:
                xi = self.pre(x[:, i])
            else:
                xi = x[:, i]
            tmp = self.model.forward_features(xi).reshape(x.size(0), 14, 14, 768).permute(0,3,1,2)
            features.append(tmp)
        return torch.stack(features, dim=1)
    
class SwinBackbone(nn.Module):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        if modality == 'flow':
            self.pre = nn.Conv2d(2, 3, 1)
        elif modality == 'flow-s5':
            self.pre = nn.Conv2d(10, 3, 1)
        self.model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
        self.model.avgpool = torch.nn.Identity()
    
    def forward(self, x):
        features = []
        for i in range(x.size(1)):
            if 'flow' in self.modality:
                xi = self.pre(x[:, i])
            else:
                xi = x[:, i]
            tmp = self.model.forward_features(xi).reshape(-1, 1024, 7, 7)
            # tmp = self.model.forward_features(xi).permute(0,2,1).reshape(-1, 1024, 7, 7)
            features.append(tmp)
        return torch.stack(features, dim=1)

class MultiScaleSwinBackbone(nn.Module):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        if modality == 'flow':
            self.pre = nn.Conv2d(2, 3, 1)
        elif modality == 'flow-s5':
            self.pre = nn.Conv2d(10, 3, 1)
        self.model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
    
    def forward(self, x):
        features = []
        for i in range(x.size(1)):
            if 'flow' in self.modality:
                xi = self.pre(x[:, i])
            else:
                xi = x[:, i]
            _, out_list = self.model.forward_features(xi, multi_scale=True)#.permute(0,2,1).reshape(-1, 1024, 7, 7)
            tmp = []
            tmp.append(F.max_pool2d(out_list[0].permute(0,2,1).reshape(-1, 256, 28, 28), 4))
            tmp.append(F.max_pool2d(out_list[1].permute(0,2,1).reshape(-1, 512, 14, 14), 2))
            tmp.append(out_list[2].permute(0,2,1).reshape(-1, 1024, 7, 7))
            tmp.append(out_list[3].permute(0,2,1).reshape(-1, 1024, 7, 7))
            tmp = torch.cat(tmp, dim=1)
            features.append(tmp)
        return torch.stack(features, dim=1)
    
class ConvNeXtBackbone(nn.Module):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        if modality == 'flow':
            self.pre = nn.Conv2d(2, 3, 1)
        elif modality == 'flow-s5':
            self.pre = nn.Conv2d(10, 3, 1)
        self.model = timm.create_model('convnext_base_in22k', pretrained=True)
    
    def forward(self, x):
        features = []
        for i in range(x.size(1)):
            if 'flow' in self.modality:
                xi = self.pre(x[:, i])
            else:
                xi = x[:, i]
            tmp = self.model.forward_features(xi)
            features.append(tmp)
        return torch.stack(features, dim=1)
    
class Convmixer_1024_20_ks9_p14_Backbone(nn.Module):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        if modality == 'flow':
            self.pre = nn.Conv2d(2, 3, 1)
        elif modality == 'flow-s5':
            self.pre = nn.Conv2d(10, 3, 1)
        self.model = timm.create_model('convmixer_1024_20_ks9_p14', pretrained=True)
    
    def forward(self, x):
        features = []
        for i in range(x.size(1)):
            if 'flow' in self.modality:
                xi = self.pre(x[:, i])
            else:
                xi = x[:, i]
            tmp = self.model.forward_features(xi)
            features.append(tmp)
        return torch.stack(features, dim=1)
    
class MixerBackbone(nn.Module):
    def __init__(self, modality):
        super().__init__()
        self.modality = modality
        if modality == 'flow':
            self.pre = nn.Conv2d(2, 3, 1)
        elif modality == 'flow-s5':
            self.pre = nn.Conv2d(10, 3, 1)
        self.model = timm.create_model('mixer_b16_224_miil_in21k', pretrained=True)
    
    def forward(self, x):
        features = []
        for i in range(x.size(1)):
            if 'flow' in self.modality:
                xi = self.pre(x[:, i])
            else:
                xi = x[:, i]
            tmp = self.model.forward_features(xi).permute(0,2,1).reshape(-1, 768, 14, 14)
            features.append(tmp)
        return torch.stack(features, dim=1)