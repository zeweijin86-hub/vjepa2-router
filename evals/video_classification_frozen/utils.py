# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.datasets.utils.video.randerase import RandomErasing


def make_transforms(
    training=True,
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3 / 4, 4 / 3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    num_views_per_clip=1,
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    color_jitter=False,
):

    if not training and num_views_per_clip > 1:
        print("Making EvalVideoTransform, multi-view")
        _frames_augmentation = EvalVideoTransform(
            num_views_per_clip=num_views_per_clip,
            short_side_size=crop_size,
            normalize=normalize,
        )

    else:
        _frames_augmentation = VideoTransform(
            training=training,
            random_horizontal_flip=random_horizontal_flip,
            random_resize_aspect_ratio=random_resize_aspect_ratio,
            random_resize_scale=random_resize_scale,
            reprob=reprob,
            auto_augment=auto_augment,
            motion_shift=motion_shift,
            crop_size=crop_size,
            normalize=normalize,
            color_jitter=(training and color_jitter),
        )
    return _frames_augmentation


class VideoTransform(object):

    def __init__(
        self,
        training=True,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3 / 4, 4 / 3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        color_jitter=False,
    ):

        self.training = training

        short_side_size = int(crop_size * 256 / 224)
        self.eval_transform = video_transforms.Compose(
            [
                video_transforms.Resize(short_side_size, interpolation="bilinear"),
                video_transforms.CenterCrop(size=(crop_size, crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=normalize[0], std=normalize[1]),
            ]
        )

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.normalize = torch.tensor(normalize)

        # Optional clip-consistent color jitter augmentation
        self.color_jitter = color_jitter
        self.jitter_transform = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        )

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment="rand-m7-n4-mstd0.5-inc1",
            interpolation="bicubic",
        )

        self.spatial_transform = (
            video_transforms.random_resized_crop_with_shift if motion_shift else video_transforms.random_resized_crop
        )

        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode="pixel",
            max_count=1,
            num_splits=1,
            device="cpu",
        )

    def _apply_clip_color_jitter(self, frames):
        """Sample ColorJitter parameters once per clip to preserve temporal consistency."""
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = (
            transforms.ColorJitter.get_params(
                self.jitter_transform.brightness,
                self.jitter_transform.contrast,
                self.jitter_transform.saturation,
                self.jitter_transform.hue,
            )
        )
        out = []
        for img in frames:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = TF.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = TF.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = TF.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = TF.adjust_hue(img, hue_factor)
            out.append(img)
        return out

    def __call__(self, buffer):

        if not self.training:
            return [self.eval_transform(buffer)]

        # Convert frames to PIL images for torchvision augmentations
        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        if self.color_jitter:
            buffer = self._apply_clip_color_jitter(buffer)

        if self.auto_augment:
            buffer = self.autoaug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        buffer = tensor_normalize(buffer, self.normalize[0], self.normalize[1])
        buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W

        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )

        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return [buffer]


class EvalVideoTransform(object):

    def __init__(
        self,
        num_views_per_clip=1,
        short_side_size=224,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ):
        self.views_per_clip = num_views_per_clip
        self.short_side_size = short_side_size
        self.spatial_resize = video_transforms.Resize(short_side_size, interpolation="bilinear")
        self.to_tensor = video_transforms.Compose(
            [
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=normalize[0], std=normalize[1]),
            ]
        )

    def __call__(self, buffer):

        # Sample several spatial views of each clip
        buffer = np.array(self.spatial_resize(buffer))
        T, H, W, C = buffer.shape

        num_views = self.views_per_clip
        side_len = self.short_side_size
        spatial_step = (max(H, W) - side_len) // (num_views - 1)

        all_views = []
        for i in range(num_views):
            start = i * spatial_step
            if H > W:
                view = buffer[:, start : start + side_len, :, :]
            else:
                view = buffer[:, :, start : start + side_len, :]
            view = self.to_tensor(view)
            all_views.append(view)

        return all_views


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if isinstance(mean, list):
        mean = torch.tensor(mean)
    if isinstance(std, list):
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
