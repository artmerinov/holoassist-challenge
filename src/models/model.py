import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
from einops.layers.torch import Rearrange
from typing import Literal


class VideoModel(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_segments: int,
                 base_model: str, 
                 fusion_mode: str,
                 verbose: bool,
                 pretrained: str,
                 mode: Literal["train", "validation", "test"] = "train",
    ) -> None:
        super(VideoModel, self).__init__()
        self.num_classes = num_classes
        self.num_segments = num_segments
        self.verbose = verbose
        self.fusion_mode = fusion_mode
        self.base_model_name = base_model
        self.pretrained = pretrained
        self.mode = mode
        
        if self.base_model_name == "TimeSformer":
            self._prepare_timesformer224()
        
        elif self.base_model_name == "InceptionV3":
            if self.pretrained == "ImageNet":
                print(f"=> Using {self.pretrained} weights")
                self._prepare_inceptionv3_pretrained_imagenet()
            elif self.pretrained == "SS1":
                print(f"=> Using {self.pretrained} weights")
                self._prepare_inceptionv3_pretrained_ss1()
            else:
                raise ValueError(f"No such pretrained model for {self.base_model_name}")
            
        elif self.base_model_name == "ResNet50":
            if self.pretrained == "ImageNet":
                print(f"=> Using {self.pretrained} weights")
                self._prepare_resnet50_pretrained_imagenet()
            elif self.pretrained == "SS1":
                print(f"=> Using {self.pretrained} weights")
                self._prepare_resnet50_pretrained_ss1()
            else:
                raise ValueError(f"No such pretrained model for {self.base_model_name}")

        else:
            raise ValueError("No such model.")

        self.print_learnable_params()

    def _prepare_timesformer224(self):
        
        import os
        os.environ['HF_HOME'] = "/data/amerinov/data/backbones/huggingface"

        import timm
        from timesformer.models.vit import TimeSformer 

        # TODO: try different VIT model
        # e.g. 384 resolution, SWIN Transformer, SS2 weights

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

    def _prepare_inceptionv3_pretrained_imagenet(self):

        if self.fusion_mode == "GSF":
            print(f"=> Using {self.fusion_mode} fusion")
            from ..backbones.pytorch_load import InceptionV3_gsf
            self.gsf_ch_ratio = 100
            self.base_model = InceptionV3_gsf(
                num_segments=self.num_segments, 
                gsf_ch_ratio=self.gsf_ch_ratio
            )
        elif self.fusion_mode == "GSM":
            print(f"=> Using {self.fusion_mode} fusion")
            from ..backbones.pytorch_load import InceptionV3_gsm
            self.base_model = InceptionV3_gsm(
                num_segments=self.num_segments
            )
        elif self.fusion_mode == "TSM":
            print(f"=> Using {self.fusion_mode} fusion")
            from ..backbones.pytorch_load import InceptionV3_tsm
            self.n_div = 8
            self.base_model = InceptionV3_tsm(
                num_segments=self.num_segments, 
                n_div=self.n_div
            )
        elif self.fusion_mode == "TSN":
            print(f"=> Using {self.fusion_mode} fusion")
            from ..backbones.pytorch_load import InceptionV3
            self.base_model = InceptionV3()
        else:
            raise ValueError(f"Incorrect fusion mode {self.fusion_mode}")
        
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

    def _prepare_inceptionv3_pretrained_ss1(self):

        if self.fusion_mode == "GSF":
            print(f"=> Using {self.fusion_mode} fusion", flush=True)
            from ..backbones.pytorch_load import InceptionV3_gsf
            self.gsf_ch_ratio = 100
            self.base_model = InceptionV3_gsf(
                num_segments=self.num_segments,
                gsf_ch_ratio=self.gsf_ch_ratio
            )
        elif self.fusion_mode == "GSM":
            print(f"=> Using {self.fusion_mode} fusion")
            from ..backbones.pytorch_load import InceptionV3_gsm
            self.base_model = InceptionV3_gsm(
                num_segments=self.num_segments,
            )
        elif self.fusion_mode == "TSM":
            print(f"=> Using {self.fusion_mode} fusion")
            from ..backbones.pytorch_load import InceptionV3_tsm
            self.n_div = 8
            self.base_model = InceptionV3_tsm(
                num_segments=self.num_segments,
                n_div=self.n_div
            )
        elif self.fusion_mode == "TSN":
            print(f"=> Using {self.fusion_mode} fusion")
            # from ..archs.inceptionv3 import inceptionv3
            # self.base_model = inceptionv3()
            from ..backbones.pytorch_load import InceptionV3
            self.base_model = InceptionV3()
        else:
            raise ValueError(f"Incorrect fusion mode {self.fusion_mode}")

        # ---------------------------------- LOAD ----------------------------------

        if self.mode == "train":

            # chk = torch.load('/data/users/amerinov/data/backbones/something-v1_inceptionv3_16frames.pth.tar')
            chk = torch.load("/Users/artemmerinov/PycharmProjects/holoassist-challenge/checkpoints/something-v1_inceptionv3_16frames.pth.tar", map_location=torch.device('cpu'))
            chk_model_state_dict = chk["model_state_dict"]

            # Create a new state dictionary with modified keys
            new_state_dict = {}
            for key in chk_model_state_dict.keys():
                new_key = key.replace("module.base_model.", "")
                new_state_dict[new_key] = chk_model_state_dict[key]

            # new_state_dict["top_cls_fc.weight"] = chk_model_state_dict["module.new_fc.weight"]
            # new_state_dict["top_cls_fc.bias"] = chk_model_state_dict["module.new_fc.bias"]

            # Count the number of keys before loading into the model
            initial_key_count = len(new_state_dict)
            print(f"Number of keys in the checkpoint: {initial_key_count}", flush=True)

            model_state = self.base_model.state_dict()
            loaded_keys_count = 0
            for key in model_state:
                if key in new_state_dict:
                    if model_state[key].shape == new_state_dict[key].shape:
                        loaded_keys_count += 1
                    else:
                        del new_state_dict[key]
                        print(f"Remove {key}", flush=True)
                else:
                    print("Missing:", key)
            print(f"Number of keys loaded into the model: {loaded_keys_count}", flush=True)

            self.base_model.load_state_dict(new_state_dict, strict=False)

        # ----------------------------------------------------------------------------
        
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

    def _prepare_resnet50_pretrained_imagenet(self):

        if self.fusion_mode == "GSF":
            print(f"=> Using {self.fusion_mode} fusion")
            from ..archs.resnet_gsf import resnet50

            self.gsf_ch_ratio = 100
            self.base_model = resnet50(
                pretrained=True,
                num_segments=self.num_segments,
                gsf_ch_ratio=self.gsf_ch_ratio
            )
        else:
            raise NotImplementedError()
        
        self.base_model.last_layer_name = 'fc'
        self.input_space = "RGB"
        self.input_size = 224
        self.input_range = [0, 1]
        self.div = True
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        
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

    def _prepare_resnet50_pretrained_ss1(self):

        if self.fusion_mode == "GSF":
            print(f"=> Using {self.fusion_mode} fusion")
            from ..archs.resnet_gsf import resnet50

            self.gsf_ch_ratio = 100
            self.base_model = resnet50(
                pretrained=True,
                num_segments=self.num_segments,
                gsf_ch_ratio=self.gsf_ch_ratio
            )
        else:
            raise NotImplementedError()
        
        # ---------------------------------- LOAD ----------------------------------

        chk = torch.load('/data/users/amerinov/data/backbones/something-v1_resnet50_16frames.pth.tar')
        chk_model_state_dict = chk["model_state_dict"]

        # Create a new state dictionary with modified keys
        new_state_dict = {}
        for key in chk_model_state_dict.keys():
            new_key = key.replace("module.base_model.", "")
            new_state_dict[new_key] = chk_model_state_dict[key]

        # new_state_dict["top_cls_fc.weight"] = chk_model_state_dict["module.new_fc.weight"]
        # new_state_dict["top_cls_fc.bias"] = chk_model_state_dict["module.new_fc.bias"]

        # Count the number of keys before loading into the model
        initial_key_count = len(new_state_dict)
        print(f"Number of keys in the checkpoint: {initial_key_count}")

        model_state = self.base_model.state_dict()
        loaded_keys_count = 0
        for key in model_state:
            if key in new_state_dict:
                if model_state[key].shape == new_state_dict[key].shape:
                    loaded_keys_count += 1
                else:
                    del new_state_dict[key]
                    print(f"Remove {key}")
            else:
                print("Missing:", key)
        print(f"Number of keys loaded into the model: {loaded_keys_count}")

        self.base_model.load_state_dict(new_state_dict, strict=False)

        # ----------------------------------------------------------------------------
        
        self.base_model.last_layer_name = 'fc'
        self.input_space = "RGB"
        self.input_size = 224
        self.input_range = [0, 1]
        self.div = True
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        
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

        elif self.base_model_name in ["ResNet50", "InceptionV3"]:
            # It is important to have (t c) and not (c t)
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
