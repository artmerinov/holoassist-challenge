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
                 fusion_mode: str,
                 dropout: float,
                 verbose: bool,
    ):
        super(VideoModel, self).__init__()
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.dropout = dropout
        self.verbose = verbose
        self.fusion_mode = fusion_mode
        self.base_model_name = base_model

        if self.base_model_name == "TimeSformer":
            self._prepare_timesformer224()
        else:
            self._prepare_base_model()
            self._prepare_model()

        self.print_learnable_params()

    def _prepare_model(self):

        # Get the number of input features of the last layer of base model.
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        
        # Replace the last layer with a dropout layer and new fully-connected layer. 
        replacement = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(feature_dim, self.num_classes)
        )
        setattr(self.base_model, self.base_model.last_layer_name, replacement)

        # Initialize weights and biases of new fully-connected layer.
        normal_(replacement[1].weight, mean=0, std=0.001)
        constant_(replacement[1].bias, val=0)

    def _prepare_timesformer224(self):
            
            import os
            os.environ['HF_HOME'] = "/data/amerinov/data/backbones/huggingface"

            import timm
            from timesformer.models.vit import TimeSformer 

            vit_model = timm.create_model(
                model_name='timm/vit_base_patch16_224.augreg_in21k_ft_in1k',
                pretrained=True,
                num_classes=self.num_classes, 
            )
            vit_state = vit_model.state_dict()

            for key in list(vit_state.keys()):
                vit_state[f"model.{key}"] = vit_state.pop(key)

            self.base_model = TimeSformer(
                img_size=224, 
                num_classes=self.num_classes, 
                num_frames=self.num_segments, 
                attention_type='divided_space_time',
            )
            self.base_model.load_state_dict(vit_state, strict=False)

            self.base_model.last_layer_name = 'classifier'
            self.input_size = 224
            self.input_space = "RGB"
            self.input_range = [0, 1]
            self.div = True
            self.input_mean = [0.5, 0.5, 0.5]
            self.input_std = [0.5, 0.5, 0.5]

    def _prepare_base_model(self):
        
        if self.base_model_name == 'BNInception':

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
            self.div = False
            self.input_mean = [104, 117, 128]
            self.input_std = [1, 1, 1]

        elif self.base_model_name == 'InceptionV3':

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
            self.div = True
            self.input_mean = [0.5, 0.5, 0.5]
            self.input_std = [0.5, 0.5, 0.5]

        else:
            raise ValueError(f"Unknown base model: {self.base_model_name}")
    
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

        # Make tensor of size requierd for base model
        if self.base_model_name == "TimeSformer":
            # https://github.com/facebookresearch/TimeSformer
            x = Rearrange("n (t c) h w -> n c t h w", n=n, c=c, t=t, h=h, w=w)(x) # b c t h w 
            logits = self.base_model(x)
            return logits
        else:
            # (!) it is important to have (t c) and not (c t)
            x = Rearrange("n (t c) h w -> (n t) c h w", n=n, c=c, t=t, h=h, w=w)(x) # [16*5, 3, 224, 224]
            x = self.base_model(x) # [16*5, num_classes]
            # Use average consensus over segments (time)
            x = Rearrange("(n t) k -> n t k", n=n, t=t, k=self.num_classes)(x) # [16, 5, num_classes]
            x = torch.mean(x, dim=1, keepdim=False) # [16, num_classes]
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
