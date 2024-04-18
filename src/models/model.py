import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
from einops.layers.torch import Rearrange


class VideoModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_segments: int,
                 base_model: str, 
                 fusion_mode: str = "GSM",
                 dropout: float = 0.5,
                 verbose: bool = True,
    ):
        super(VideoModel, self).__init__()
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.dropout = dropout
        self.verbose = verbose
        self.fusion_mode = fusion_mode

        self._prepare_base_model(base_model)
        self._prepare_model(num_classes)
        self.print_learnable_params()

    def _prepare_model(self, num_classes: int):

        # Get the number of input features of the last layer of base model.
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        
        if self.dropout == 0:
            # Replace the last layer of the base model with a new fully-connected layer.
            replacement = nn.Linear(feature_dim, num_classes)
            setattr(self.base_model, self.base_model.last_layer_name, replacement)
            
            # Initialize weights and biases of new fully-connected layer.
            normal_(replacement.weight, mean=0, std=0.001)
            constant_(replacement.bias, val=0)
        else:
            # Replace the last layer of the base model with a dropout layer 
            # and new fully-connected layer after the dropout layer. 
            replacement = nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(feature_dim, num_classes)
            )
            setattr(self.base_model, self.base_model.last_layer_name, replacement)

            # Initialize weights and biases of new fully-connected layer.
            normal_(replacement[1].weight, mean=0, std=0.001)
            constant_(replacement[1].bias, val=0)

    def _prepare_base_model(self, base_model: str):
        
        if base_model == 'BNInception':

            if self.fusion_mode == "GSM":
                print(f"=> Using {self.fusion_mode} fusion")
                from ..archs.bn_inception_gsm import bninception
                
                self.gsf_ch_ratio = 100
                self.base_model = bninception(
                    num_segments=self.num_segments, 
                    gsf_ch_ratio=self.gsf_ch_ratio
                )
            else:
                print(f"=> Using {self.fusion_mode} fusion")
                from ..archs.bn_inception import bninception
                
                self.base_model = bninception()

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_space = "BGR"
            self.input_range = [0, 255]
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]

        elif base_model == 'InceptionV3':

            if self.fusion_mode == "GSF":
                print(f"=> Using {self.fusion_mode} fusion")
                from ..backbones.pytorch_load import InceptionV3_gsf
                self.base_model = InceptionV3_gsf(
                    num_segments=self.num_segments,
                    gsf_ch_ratio=100
                )
            elif self.fusion_mode == "GSM":
                raise NotImplementedError()
                # print(f"=> Using {self.fusion_mode} fusion")
                # from ..backbones.pytorch_load import InceptionV3_gsm
                # self.base_model = InceptionV3_gsm(
                #     num_segments=self.num_segments,
                # )
            else:
                print(f"=> Using {self.fusion_mode} fusion")
                from ..archs.inceptionv3 import inceptionv3
                self.base_model = inceptionv3()
                # from ..backbones.pytorch_load import InceptionV3
                # self.base_model = InceptionV3()
            
            # self.base_model.last_layer_name = 'fc'
            self.base_model.last_layer_name = 'top_cls_fc'
            self.input_size = 299
            self.input_space = "RGB"
            self.input_range = [0, 1]
            self.input_mean = [0.5, 0.5, 0.5]
            self.input_std = [0.5, 0.5, 0.5]

        else:
            raise ValueError(f"Unknown base model: {base_model}")
    
    def forward(self, x):
        
        # After stack a group of images along channel dimension, 
        # we have input tensor of size (n, t*c, h, w), where
        # n -- batch size
        # t -- time (number of sampled segments)
        # c -- number of channels
        # h -- hight
        # w -- width

        n, t_c, h, w = x.shape # [16, 5*3, 224, 224]
        t = self.num_segments
        c = t_c // t

        # Make tensor of size requierd for base model [16*5, 3, 224, 224]
        # (!) it is important to have (t c) and not (c t)
        # and apply base model which outputs the scores distribution of classes
        x = Rearrange("n (t c) h w -> (n t) c h w", n=n, c=c, t=t, h=h, w=w)(x) # [16*5, 3, 224, 224]
        x = self.base_model(x) # [16*5, 101]
        
        # Use average consensus over segments (time)
        x = Rearrange("(n t) k -> n t k", n=n, t=t, k=self.num_classes)(x) # [16, 5, 101]
        x = torch.mean(x, dim=1, keepdim=False) # [16, 101]

        return x

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def print_learnable_params(self):
        self.learnable_named_parameters = [
            (name, p) for name, p in self.named_parameters() if p.requires_grad
        ]

        if self.verbose:
            # Print all learable parameters
            for name, p in self.learnable_named_parameters:
                print(f"#params: {p.numel()} {name:<10}", flush=True)

            total_params = sum(p.numel() for _, p in self.learnable_named_parameters)
            print(f'Total number of learnable parameters: {total_params}', flush=True)
