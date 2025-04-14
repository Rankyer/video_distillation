import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
from data.dataset_statistics import STDS, MEANS
from data.augment import Normalize
import torchvision.transforms as transforms


class DiffAug:
    def __init__(
        self,
        strategy="color_crop_cutout_flip_scale_rotate",
        batch=False,
        ratio_cutout=0.5,
        single=False,
    ):
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = ratio_cutout
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5

        self.batch = batch

        self.aug = True
        if strategy == "" or strategy.lower() == "none":
            self.aug = False
        else:
            self.strategy = []
            self.flip = False
            self.color = False
            self.cutout = False
            for aug in strategy.lower().split("_"):
                if aug == "flip" and single == False:
                    self.flip = True
                elif aug == "color" and single == False:
                    self.color = True
                elif aug == "cutout" and single == False:
                    self.cutout = True
                else:
                    self.strategy.append(aug)

        self.aug_fn = {
            "color": [self.brightness_fn, self.saturation_fn, self.contrast_fn],
            "crop": [self.crop_fn],
            "cutout": [self.cutout_fn],
            "flip": [self.flip_fn],
            "scale": [self.scale_fn],
            "rotate": [self.rotate_fn],
            "translate": [self.translate_fn],
        }

    def __call__(self, x, single_aug=True, seed=-1):
        if not self.aug:
            return x
        else:
            if self.flip:
                self.set_seed(seed)
                x = self.flip_fn(x, self.batch)
            if self.color:
                for f in self.aug_fn["color"]:
                    self.set_seed(seed)
                    x = f(x, self.batch)
            if len(self.strategy) > 0:
                if single_aug:
                    # single
                    idx = np.random.randint(len(self.strategy))
                    p = self.strategy[idx]
                    for f in self.aug_fn[p]:
                        self.set_seed(seed)
                        x = f(x, self.batch)
                else:
                    # multiple
                    for p in self.strategy:
                        for f in self.aug_fn[p]:
                            self.set_seed(seed)
                            x = f(x, self.batch)
            if self.cutout:
                self.set_seed(seed)
                x = self.cutout_fn(x, self.batch)

            x = x.contiguous()
            return x

    def set_seed(self, seed):
        if seed > 0:
            np.random.seed(seed)
            torch.random.manual_seed(seed)

    # def scale_fn(self, x, batch=True):
    #     # x>1, max scale
    #     # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    #     ratio = self.ratio_scale

    #     if batch:
    #         sx = np.random.uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
    #         sy = np.random.uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
    #         theta = [[sx, 0, 0], [0, sy, 0]]
    #         theta = torch.tensor(theta, dtype=torch.float, device=x.device)
    #         theta = theta.expand(x.shape[0], 2, 3)
    #     else:
    #         sx = (
    #             np.random.uniform(size=x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    #         )
    #         sy = (
    #             np.random.uniform(size=x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    #         )
    #         theta = [[[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0])]
    #         theta = torch.tensor(theta, dtype=torch.float, device=x.device)

    #     grid = F.affine_grid(theta, x.shape)
    #     x = F.grid_sample(x, grid)
    #     return x

    # def rotate_fn(self, x, batch=True):
    #     # [-180, 180], 90: anticlockwise 90 degree
    #     ratio = self.ratio_rotate

    #     if batch:
    #         theta = (np.random.uniform() - 0.5) * 2 * ratio / 180 * float(np.pi)
    #         theta = [
    #             [np.cos(theta), np.sin(-theta), 0],
    #             [np.sin(theta), np.cos(theta), 0],
    #         ]
    #         theta = torch.tensor(theta, dtype=torch.float, device=x.device)
    #         theta = theta.expand(x.shape[0], 2, 3)
    #     else:
    #         theta = (
    #             (np.random.uniform(size=x.shape[0]) - 0.5)
    #             * 2
    #             * ratio
    #             / 180
    #             * float(np.pi)
    #         )
    #         theta = [
    #             [
    #                 [np.cos(theta[i]), np.sin(-theta[i]), 0],
    #                 [np.sin(theta[i]), np.cos(theta[i]), 0],
    #             ]
    #             for i in range(x.shape[0])
    #         ]
    #         theta = torch.tensor(theta, dtype=torch.float, device=x.device)

    #     grid = F.affine_grid(theta, x.shape)
    #     x = F.grid_sample(x, grid)
    #     return x

    # def flip_fn(self, x, batch=True):
    #     prob = self.prob_flip

    #     if batch:
    #         coin = np.random.uniform()
    #         if coin < prob:
    #             return x.flip(3)
    #         else:
    #             return x
    #     else:
    #         randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    #         return torch.where(randf < prob, x.flip(3), x)

    # def brightness_fn(self, x, batch=True):
    #     # mean
    #     ratio = self.brightness

    #     if batch:
    #         randb = np.random.uniform()
    #     else:
    #         randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    #     x = x + (randb - 0.5) * ratio
    #     return x

    # def saturation_fn(self, x, batch=True):
    #     # channel concentration
    #     ratio = self.saturation

    #     x_mean = x.mean(dim=1, keepdim=True)
    #     if batch:
    #         rands = np.random.uniform()
    #     else:
    #         rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    #     x = (x - x_mean) * (rands * ratio) + x_mean
    #     return x

    # def contrast_fn(self, x, batch=True):
    #     # spatially concentrating
    #     ratio = self.contrast

    #     x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    #     if batch:
    #         randc = np.random.uniform()
    #     else:
    #         randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    #     x = (x - x_mean) * (randc + ratio) + x_mean
    #     return x

    # def translate_fn(self, x, batch=True):
    #     ratio = self.ratio_crop_pad

    #     shift_y = int(x.size(3) * ratio + 0.5)
    #     if batch:
    #         translation_y = np.random.randint(-shift_y, shift_y + 1)
    #     else:
    #         translation_y = torch.randint(
    #             -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
    #         )

    #     grid_batch, grid_x, grid_y = torch.meshgrid(
    #         torch.arange(x.size(0), dtype=torch.long, device=x.device),
    #         torch.arange(x.size(2), dtype=torch.long, device=x.device),
    #         torch.arange(x.size(3), dtype=torch.long, device=x.device),
    #     )
    #     grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    #     x_pad = F.pad(x, (1, 1))
    #     x = (
    #         x_pad.permute(0, 2, 3, 1)
    #         .contiguous()[grid_batch, grid_x, grid_y]
    #         .permute(0, 3, 1, 2)
    #     )
    #     return x

    # def crop_fn(self, x, batch=True):
    #     # The image is padded on its surrounding and then cropped.
    #     ratio = self.ratio_crop_pad

    #     shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    #     if batch:
    #         translation_x = np.random.randint(-shift_x, shift_x + 1)
    #         translation_y = np.random.randint(-shift_y, shift_y + 1)
    #     else:
    #         translation_x = torch.randint(
    #             -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
    #         )

    #         translation_y = torch.randint(
    #             -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
    #         )

    #     grid_batch, grid_x, grid_y = torch.meshgrid(
    #         torch.arange(x.size(0), dtype=torch.long, device=x.device),
    #         torch.arange(x.size(2), dtype=torch.long, device=x.device),
    #         torch.arange(x.size(3), dtype=torch.long, device=x.device),
    #     )
    #     grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    #     grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    #     x_pad = F.pad(x, (1, 1, 1, 1))
    #     x = (
    #         x_pad.permute(0, 2, 3, 1)
    #         .contiguous()[grid_batch, grid_x, grid_y]
    #         .permute(0, 3, 1, 2)
    #     )
    #     return x

    # def cutout_fn(self, x, batch=True):
    #     ratio = self.ratio_cutout
    #     cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

    #     if batch:
    #         offset_x = np.random.randint(0, x.size(2) + (1 - cutout_size[0] % 2))
    #         offset_y = np.random.randint(0, x.size(3) + (1 - cutout_size[1] % 2))
    #     else:
    #         offset_x = torch.randint(
    #             0,
    #             x.size(2) + (1 - cutout_size[0] % 2),
    #             size=[x.size(0), 1, 1],
    #             device=x.device,
    #         )

    #         offset_y = torch.randint(
    #             0,
    #             x.size(3) + (1 - cutout_size[1] % 2),
    #             size=[x.size(0), 1, 1],
    #             device=x.device,
    #         )

    #     grid_batch, grid_x, grid_y = torch.meshgrid(
    #         torch.arange(x.size(0), dtype=torch.long, device=x.device),
    #         torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
    #         torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    #     )
    #     grid_x = torch.clamp(
    #         grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1
    #     )
    #     grid_y = torch.clamp(
    #         grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1
    #     )
    #     mask = torch.ones(
    #         x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device
    #     )
    #     mask[grid_batch, grid_x, grid_y] = 0
    #     x = x * mask.unsqueeze(1)
    #     return x


    def scale_fn(self, x, batch=True):
        ratio = self.ratio_scale

        if batch:
            sx = np.random.uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
            sy = np.random.uniform() * (ratio - 1.0 / ratio) + 1.0 / ratio
            theta = [[sx, 0, 0], [0, sy, 0]]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)
            theta = theta.expand(x.shape[0] * x.shape[1], 2, 3)  # batch * frames
        else:
            sx = (
                np.random.uniform(size=x.shape[0] * x.shape[1]) * (ratio - 1.0 / ratio) + 1.0 / ratio
            )
            sy = (
                np.random.uniform(size=x.shape[0] * x.shape[1]) * (ratio - 1.0 / ratio) + 1.0 / ratio
            )
            theta = [[[sx[i], 0, 0], [0, sy[i], 0]] for i in range(x.shape[0] * x.shape[1])]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)

        x_reshaped = x.reshape(-1, *x.shape[2:])  # 合并 batch 和 frames
        grid = F.affine_grid(theta, x_reshaped.shape, align_corners=False)
        x_scaled = F.grid_sample(x_reshaped, grid, align_corners=False)
        return x_scaled.reshape(x.shape)


    def rotate_fn(self, x, batch=True):
        ratio = self.ratio_rotate

        if batch:
            theta = (np.random.uniform() - 0.5) * 2 * ratio / 180 * float(np.pi)
            theta = [
                [np.cos(theta), np.sin(-theta), 0],
                [np.sin(theta), np.cos(theta), 0],
            ]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)
            theta = theta.expand(x.shape[0] * x.shape[1], 2, 3)  # batch * frames
        else:
            theta = (
                (np.random.uniform(size=x.shape[0] * x.shape[1]) - 0.5)
                * 2
                * ratio
                / 180
                * float(np.pi)
            )
            theta = [
                [
                    [np.cos(theta[i]), np.sin(-theta[i]), 0],
                    [np.sin(theta[i]), np.cos(theta[i]), 0],
                ]
                for i in range(x.shape[0] * x.shape[1])
            ]
            theta = torch.tensor(theta, dtype=torch.float, device=x.device)

        x_reshaped = x.reshape(-1, *x.shape[2:])  # 合并 batch 和 frames
        grid = F.affine_grid(theta, x_reshaped.shape, align_corners=False)
        x_rotated = F.grid_sample(x_reshaped, grid, align_corners=False)
        return x_rotated.reshape(x.shape)


    def flip_fn(self, x, batch=True):
        prob = self.prob_flip

        if batch:
            coin = np.random.uniform()
            if coin < prob:
                return x.flip(4)  # 水平翻转，flip 宽度维度
            else:
                return x
        else:
            randf = torch.rand(x.size(0), x.size(1), 1, 1, 1, device=x.device)
            return torch.where(randf < prob, x.flip(4), x)


    def brightness_fn(self, x, batch=True):
        ratio = self.brightness

        if batch:
            randb = np.random.uniform()
        else:
            randb = torch.rand(x.size(0), x.size(1), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = x + (randb - 0.5) * ratio
        return x


    def saturation_fn(self, x, batch=True):
        ratio = self.saturation

        x_mean = x.mean(dim=2, keepdim=True)  # 对通道求均值
        if batch:
            rands = np.random.uniform()
        else:
            rands = torch.rand(x.size(0), x.size(1), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (rands * ratio) + x_mean
        return x


    def contrast_fn(self, x, batch=True):
        ratio = self.contrast

        x_mean = x.mean(dim=[2, 3, 4], keepdim=True)  # 对空间和通道维度求均值
        if batch:
            randc = np.random.uniform()
        else:
            randc = torch.rand(x.size(0), x.size(1), 1, 1, 1, dtype=x.dtype, device=x.device)
        x = (x - x_mean) * (randc + ratio) + x_mean
        return x


    def translate_fn(self, x, batch=True):
        ratio = self.ratio_crop_pad

        shift_y = int(x.size(3) * ratio + 0.5)
        shift_x = int(x.size(4) * ratio + 0.5)
        if batch:
            translation_y = np.random.randint(-shift_y, shift_y + 1)
            translation_x = np.random.randint(-shift_x, shift_x + 1)
        else:
            translation_y = torch.randint(
                -shift_y, shift_y + 1, size=[x.size(0), x.size(1), 1, 1], device=x.device
            )
            translation_x = torch.randint(
                -shift_x, shift_x + 1, size=[x.size(0), x.size(1), 1, 1], device=x.device
            )

        grid_batch, grid_frames, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(1), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
            torch.arange(x.size(4), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(3) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(4) + 1)
        x_pad = F.pad(x, (1, 1, 1, 1))
        x = (
            x_pad.permute(0, 1, 3, 4, 2)
            .contiguous()[grid_batch, grid_frames, grid_x, grid_y]
            .permute(0, 1, 4, 2, 3)
        )
        return x

    def crop_fn(self, x, batch=True):
        # x 的形状是 [batch, frames, channel, height, width]
        ratio = self.ratio_crop_pad

        shift_x, shift_y = int(x.size(3) * ratio + 0.5), int(x.size(4) * ratio + 0.5)  # 高度和宽度
        if batch:
            translation_x = np.random.randint(-shift_x, shift_x + 1)  # 水平方向平移
            translation_y = np.random.randint(-shift_y, shift_y + 1)  # 垂直方向平移
        else:
            translation_x = torch.randint(
                -shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device
            )
            translation_y = torch.randint(
                -shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device
            )

        # 创建网格：保留时间维度 frames
        grid_batch, grid_frames, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),  # batch
            torch.arange(x.size(1), dtype=torch.long, device=x.device),  # frames
            torch.arange(x.size(3), dtype=torch.long, device=x.device),  # height
            torch.arange(x.size(4), dtype=torch.long, device=x.device),  # width
        )

        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(3) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(4) + 1)

        # 填充边界：保留时间维度 frames
        x_pad = F.pad(x, (1, 1, 1, 1))  # 填充 height 和 width 维度
        x = (
            x_pad.permute(0, 1, 3, 4, 2)  # 调整维度顺序：batch, frames, height, width, channel
            .contiguous()[grid_batch, grid_frames, grid_x, grid_y]
            .permute(0, 1, 4, 2, 3)  # 恢复维度顺序：batch, frames, channel, height, width
        )
        return x



    def cutout_fn(self, x, batch=True):
        ratio = self.ratio_cutout
        cutout_size = int(x.size(3) * ratio + 0.5), int(x.size(4) * ratio + 0.5)

        if batch:
            offset_x = np.random.randint(0, x.size(3) + (1 - cutout_size[0] % 2))
            offset_y = np.random.randint(0, x.size(4) + (1 - cutout_size[1] % 2))
        else:
            offset_x = torch.randint(
                0,
                x.size(3) + (1 - cutout_size[0] % 2),
                size=[x.size(0), x.size(1), 1, 1],
                device=x.device,
            )
            offset_y = torch.randint(
                0,
                x.size(4) + (1 - cutout_size[1] % 2),
                size=[x.size(0), x.size(1), 1, 1],
                device=x.device,
            )

        grid_batch, grid_frames, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(1), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(
            grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(3) - 1
        )
        grid_y = torch.clamp(
            grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(4) - 1
        )
        mask = torch.ones(
            x.size(0), x.size(1), x.size(3), x.size(4), dtype=x.dtype, device=x.device
        )
        mask[grid_batch, grid_frames, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(2)
        return x




    def cutout_inv_fn(self, x, batch=True):
        ratio = self.ratio_cutout
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)

        if batch:
            offset_x = np.random.randint(0, x.size(2) - cutout_size[0])
            offset_y = np.random.randint(0, x.size(3) - cutout_size[1])
        else:
            offset_x = torch.randint(
                0, x.size(2) - cutout_size[0], size=[x.size(0), 1, 1], device=x.device
            )
            offset_y = torch.randint(
                0, x.size(3) - cutout_size[1], size=[x.size(0), 1, 1], device=x.device
            )

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y, min=0, max=x.size(3) - 1)
        mask = torch.zeros(
            x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device
        )
        mask[grid_batch, grid_x, grid_y] = 1.0
        x = x * mask.unsqueeze(1)
        return x


def remove_aug(augtype, remove_aug):
    aug_list = []
    for aug in augtype.split("_"):
        if aug not in remove_aug.split("_"):
            aug_list.append(aug)

    return "_".join(aug_list)


def diffaug(args, device="cuda"):
    """Differentiable augmentation for condensation"""
    aug_type = args.aug_type
    normalize = Normalize(
        mean=MEANS["imagenet"], std=STDS["imagenet"], device=device
    )
    # if args.rank == 0:
    #     print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    if args.mixup == "cut":
        aug_type = remove_aug(aug_type, "cutout")
    # if args.rank == 0:
    #     print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand


def normaug(args, device="cuda"):
    """Differentiable augmentation for condensation"""
    normalize = Normalize(
        mean=MEANS[args.dataset], std=STDS[args.dataset], device=device
    )
    norm_aug = transforms.Compose([normalize])
    return norm_aug
