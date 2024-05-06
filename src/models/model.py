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
                 verbose: bool,
    ):
        super(VideoModel, self).__init__()
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.verbose = verbose
        self.fusion_mode = fusion_mode
        self.base_model_name = base_model

        if self.base_model_name == "TimeSformer":
            self._prepare_timesformer224()
        elif self.base_model_name == "HORST":
            self._prepare_horst()
        elif self.base_model_name == "InceptionV3":
            self._prepare_inceptionv3()
        else:
            raise ValueError()

        self.print_learnable_params()

    def _prepare_timesformer224(self):
        
        import os
        os.environ['HF_HOME'] = "/data/amerinov/data/backbones/huggingface"

        import timm
        from timesformer.models.vit import TimeSformer 

        # TODO: try different VIT model
        # e.g. 384 resolution or SWIN Transformer

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

        self.input_size = 224
        self.input_space = "RGB"
        self.input_range = [0, 1]
        self.div = True
        self.input_mean = [0.5, 0.5, 0.5]
        self.input_std = [0.5, 0.5, 0.5]

    def _prepare_horst(self):
        
        from src.horst.models import AttentionRNN

        self.base_model = AttentionRNN(
            num_a=self.num_classes, 
            pretrain=False, 
            backbone="swin-base"
        )
         
        # TODO: add file path to config

        chk = torch.load('/data/amerinov/data/backbones/FAttentionRNN-anticipation_0.25_6_8_rgb_mt5r_best.pth.tar')
        # chk = torch.load('/Users/artemmerinov/data/backbones/FAttentionRNN-anticipation_0.25_6_8_rgb_mt5r_best.pth.tar',
        #                  map_location=torch.device('cpu'))
        model_state = self.base_model.state_dict()
        for i in model_state:
            if i in chk['state_dict'] and model_state[i].shape != chk['state_dict'][i].shape:
                del chk['state_dict'][i]
        self.base_model.load_state_dict(chk['state_dict'], strict=False)

        self.input_size = 224
        self.input_space = "RGB"
        self.input_range = [0, 1]
        self.div = True
        self.input_mean = [0.5, 0.5, 0.5]
        self.input_std = [0.5, 0.5, 0.5]

    def _prepare_inceptionv3(self):

        if self.fusion_mode == "GSF":
            print(f"=> Using {self.fusion_mode} fusion")
            from ..backbones.pytorch_load import InceptionV3_gsf

            self.gsf_ch_ratio = 100
            self.base_model = InceptionV3_gsf(
                num_segments=self.num_segments,
                gsf_ch_ratio=self.gsf_ch_ratio
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

        # Get the number of input features of the last layer of base model.
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        
        # Replace the last layer with a dropout layer and new fully-connected layer. 
        self.dropout = 0.5
        replacement = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(feature_dim, self.num_classes)
        )
        setattr(self.base_model, self.base_model.last_layer_name, replacement)
        
        # Initialize weights and biases of new fully-connected layer.
        normal_(replacement[1].weight, mean=0, std=0.001)
        constant_(replacement[1].bias, val=0)
    
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
        
        if self.base_model_name == "TimeSformer":
            # https://github.com/facebookresearch/TimeSformer
            x = Rearrange("n (t c) h w -> n c t h w", n=n, c=c, t=t, h=h, w=w)(x) # n c t h w
            logits = self.base_model(x)
            return logits
        
        elif self.base_model_name == "HORST":
            x = Rearrange("n (t c) h w -> n t c h w", n=n, c=c, t=t, h=h, w=w)(x) # n t c h w 
            frame_logits = self.base_model(x) # n t num_classes
            # clip_logits = torch.mean(frame_logits, dim=1, keepdim=False) # n num_classes
            # Take embedding of the last frame of the clip
            clip_logits = frame_logits[:, -1, :]
            return clip_logits
        
        elif self.base_model_name == "InceptionV3":
            # (!) it is important to have (t c) and not (c t)
            x = Rearrange("n (t c) h w -> (n t) c h w", n=n, c=c, t=t, h=h, w=w)(x) # [16*5, 3, 224, 224]
            x = self.base_model(x) # [16*5, num_classes]
            # Use average consensus over segments (time)
            x = Rearrange("(n t) k -> n t k", n=n, t=t, k=self.num_classes)(x) # [16, 5, num_classes]
            x = torch.mean(x, dim=1, keepdim=False) # [16, num_classes]
            return x

        else:
            raise ValueError()

    @property
    def crop_size(self):
        return self.input_size

    def print_learnable_params(self):
        self.learnable_named_parameters = [
            (name, p) for name, p in self.named_parameters() if p.requires_grad
        ]

        if self.verbose:
            # Print all learnable parameters
            for name, p in self.learnable_named_parameters:
                print(f"#params: {p.numel()} {name:<10}", flush=True)

            total_params = sum(p.numel() for _, p in self.learnable_named_parameters)
            print(f'Total number of learnable parameters: {total_params}', flush=True)
