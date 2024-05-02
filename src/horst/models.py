from torch import nn
import torch
import torch.nn.functional as F

from .models_backbone import ViTBackbone, SwinBackbone, ConvNeXtBackbone, Convmixer_1024_20_ks9_p14_Backbone, MixerBackbone, MultiScaleSwinBackbone
from .models_recurrent_modified import ConvLSTM
from .models_classifier import Classifier

BACKBONE_OPTIONS = {
    'vit-base': (ViTBackbone, 768),
    'swin-base': (SwinBackbone, 1024),
    'multiscale-swin-base': (MultiScaleSwinBackbone, 2816),
    'convnext': (ConvNeXtBackbone, 1024),
    'mixer': (MixerBackbone, 768),
    'convmixer': (Convmixer_1024_20_ks9_p14_Backbone, 1024),
}

class AttentionRNN(nn.Module):
    def __init__(self, num_a, num_n=None, num_v=None, pretrain=False, modality='rgb', backbone='swin-base'):
        super().__init__()
        self.backbone = backbone
        self.backbone_model = BACKBONE_OPTIONS[backbone][0](modality)
        self.norm = nn.LayerNorm(BACKBONE_OPTIONS[backbone][1])
        self.recurrent_model = nn.Sequential(
            ConvLSTM(BACKBONE_OPTIONS[backbone][1], [1, 1], [1024, 1024], [1, 1], output_ht=[True, True], skip_stride = None, cell_params = {'order': 8, 'steps': 8, 'ranks': 1024}, input_shortcut=False),
            Classifier(1024, num_a, num_n, num_v)
        )
        self.pretrain = pretrain
    
    def forward(self, x):
        if not self.pretrain:
            with torch.no_grad():
                self.backbone_model.eval()
                features = self.backbone_model(x) # B, T, C, H, W
            
            if self.backbone == 'multiscale-swin-base':
                features = self.norm(features.reshape(features.size(0)*features.size(1), 2816, -1).transpose(-1, -2)).transpose(-1, -2).reshape(*features.shape)
            
        else:
            features = self.backbone_model(x)
        return self.recurrent_model(features)