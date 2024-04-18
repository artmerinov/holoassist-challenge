import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import torch
from torchvision.transforms import functional
from typing import List, Tuple, Union, Optional
import cv2


class IdentityTransform:
    def __call__(self, data):
        return data


class GroupCenterCrop:
    """
    Center crop a group of images (video).

    Args:
        size: The size of images after center crop. If int, it is the
            size of the smaller edge, and if tuple, it is (height, width).
    """
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        self.worker = torchvision.transforms.CenterCrop(size=size)

    def __call__(self, img_group: List[Image.Image]) -> List[Image.Image]:

        cropped_imgs = []
        for img in img_group:
            cropped_img = self.worker(img)
            cropped_imgs.append(cropped_img)

        return cropped_imgs
    

class GroupRandomCrop:
    """
    Randomly crop a group of images (video).

    Args:
        size: The size of images after random crop. If int, it is the
            size of the smaller edge, and if tuple, it is (height, width).
    """
    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_group: List[Image.Image]) -> List[Image.Image]:

        w, h = img_group[0].size # (width, hight) for PIL Image
        th, tw = self.size # (height, width)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        cropped_imgs = []
        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                cropped_imgs.append(img)
            else:
                cropped_imgs.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return cropped_imgs
    

class GroupRandomHorizontalFlip:
    """
    Horizontally flips group of images with a given probability.

    Args:
        probability: Flip probability.
    """
    def __init__(self, probability: float = 0.5) -> None:
        self.probability = probability

    def __call__(self, img_group: List[Image.Image]) -> List[Image.Image]:
        if np.random.uniform(low=0, high=1) < self.probability:
            flipped_imgs = []
            for img in img_group:
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_imgs.append(flipped_img)
            return flipped_imgs
        else:
            return img_group


class GroupNormalize:
    """
    Normalize a group of input torch.Tensor images.

    Args:
        mean: Mean value for normalization.
        std: Standard deviation value for normalization.
    """
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # rep_mean is [meanR, meanG, meanB, meanR, meanG, meanB, ...]
        # rep_mean is [stdR, stdG, stdB, stdR, stdG, stdB, ...]
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor
    

class GroupScale:
    """
    Rescale a group of images to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size).
    
    Args:
        size: Size of the smaller edge.
        interpolation: Interpolation method for resizing. 
    """
    def __init__(self, size: int, interpolation: int = cv2.INTER_LINEAR) -> None:
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group: List[Image.Image]) -> List[Image.Image]:

        scaled_imgs = []
        for img in img_group:
            scales_img = self.worker(img)
            scaled_imgs.append(scales_img)

        return scaled_imgs


class GroupMultiScaleCrop:
    def __init__(
            self, 
            input_size: Union[int, Tuple[int, int]],
            scales: Optional[Tuple[float]] = [1, .875, .75, .66],
            max_distort: int = 1, 
            fix_crop: bool = True, 
            more_fix_crop: bool = True,
            interpolation: int = cv2.INTER_LINEAR,
        ) -> None:

        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            self.input_size = input_size

        self.scales = scales
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.interpolation = interpolation

    def __call__(self, img_group):
        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1] # (width, hight) for PIL Image

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(
                x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [
            self.input_size[0] if abs(
                x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRotate:
    """
    Spatially rotate group of images (video).
    
    Args:
        angle_range: Angle range boundaries.
        interpolation: Interpolation method.
        fill: Fill value.
    """
    def __init__(
            self,
            angle_range: Union[Tuple[float, float], List[float], float],
            interpolation: int = cv2.INTER_NEAREST,
            fill: float = 0.0,
            probability: float = 0.5,
            distr: str = "normal",
        ) -> List[Image.Image]:

        if isinstance(angle_range, int):
            angle_range = (-angle_range, angle_range)
        
        self.angle_range = angle_range
        self.interpolation = interpolation
        self.fill = fill
        self.probability = probability
        self.distr = distr

    def __call__(self, imgs: List[Image.Image]) -> List[Image.Image]:

        if np.random.uniform(low=0, high=1) < self.probability:

            angle_min = self.angle_range[0]
            angle_max = self.angle_range[1]
            angle_avg = (angle_min + angle_max) / 2
            angle_std = (angle_max - angle_min) / 4

            if self.distr == "uniform":
                angle = np.random.uniform(low=angle_min, high=angle_max)
            elif self.distr == "normal":
                angle = np.random.normal(loc=angle_avg, scale=angle_std)
                angle = np.clip(angle, angle_min, angle_max)
            else:
                raise ValueError()
            
            rotated_imgs = []
            for img in imgs:
                rotated_img = functional.rotate(
                    img=img,
                    angle=angle,
                    interpolation=self.interpolation,
                    expand=False,
                    center=None,
                    fill=self.fill,
                )
                rotated_imgs.append(rotated_img)
            
            return rotated_imgs
        
        else:
            return imgs


class Stack:
    """
    Stack a group of images along channel dimension 
    (H x W x C'), where C' = T*C.
    """
    def __init__(self, roll: bool = False) -> None:
        self.roll = roll

    def __call__(self, img_group: List[Image.Image]) -> np.ndarray:
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor:
    """ 
    Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 

    Args:
        div: If True, normalize pixel values by dividing by 255.
    """
    def __init__(self, div: bool = True) -> None:
        self.div = div

    def __call__(self, pic: Union[np.ndarray, Image.Image]) -> torch.FloatTensor:
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.to(torch.float32).div(255) if self.div else img.to(torch.float32)
