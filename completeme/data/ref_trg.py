import json
import random
import os
from os.path import join as osp
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def make_mask(images, resolution, times=30):
    mask, times = torch.ones_like(images[0:1, :, :]), np.random.randint(1, times)
    min_size, max_size, margin = np.array([0.03, 0.25, 0.01]) * resolution
    max_size = min(max_size, resolution - margin * 2)

    for _ in range(times):
        width = np.random.randint(int(min_size), int(max_size))
        height = np.random.randint(int(min_size), int(max_size))

        x_start = np.random.randint(int(margin), resolution - int(margin) - width + 1)
        y_start = np.random.randint(int(margin), resolution - int(margin) - height + 1)
        mask[:, y_start:y_start + height, x_start:x_start + width] = 0

    mask = 1 - mask if random.random() < 0.5 else mask
    
    return mask


class RefTrgDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        jsonl_path: str,
        image_wh: Union[int, Tuple] = (768, 768),
        num_samples: Optional[int] = None,
        load_rgba: bool = False,
        bg_color: Union[str, float] = "gray",
    ):
        super().__init__()

        with open(jsonl_path, "r") as f:
            data = [json.loads(line) for line in f if line.strip()]

        if num_samples is not None:
            data = data[:num_samples]

        image_wh = (
            (image_wh, image_wh) if isinstance(image_wh, int) else tuple(image_wh)
        )
        bg_color = (
            torch.from_numpy(self.get_bg_color(bg_color))
            if load_rgba and bg_color is not None
            else None
        )

        self.random_crop_resize = transforms.RandomResizedCrop(
            image_wh,
            scale=(0.9, 1.0),
            ratio=(0.9, 1.0),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

        # Get the inpaint mask files
        inpaint_mask_root_dir = "data/DeepFashion-MultiModal-Parts2Whole/binary_masks"
        inpaint_mask_files = [osp(inpaint_mask_root_dir, file) for file in os.listdir(inpaint_mask_root_dir)]
        self.inpaint_mask_files = inpaint_mask_files

        self.data = data
        self.root_dir = root_dir
        self.image_wh = image_wh
        self.load_rgba = load_rgba
        self.bg_color = bg_color

    def __len__(self) -> int:
        return len(self.data)

    def get_bg_color(self, bg_color):
        if bg_color == "white":
            bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        elif bg_color == "black":
            bg_color = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif bg_color == "gray":
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif bg_color == "random":
            bg_color = np.random.rand(3)
        elif bg_color == "random_gray":
            bg_color = random.uniform(0.3, 0.7)
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        elif isinstance(bg_color, float):
            bg_color = np.array([bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def load_image(
        self,
        img_path: str,
        bg_color: Optional[torch.Tensor] = None,
        rescale: bool = True,
        permute: bool = True,
    ):
        img = np.array(self.random_crop_resize(Image.open(img_path).convert("RGBA")))
        img = torch.from_numpy((img.astype(np.float32) / 255.0))  # [0, 1]

        mask = img[..., 3:4]
        img = (
            img[..., :3] * mask + bg_color * (1 - mask)
            if bg_color is not None
            else img[..., :3]
        )

        if rescale:
            img = img * 2.0 - 1.0  # to -1 ~ 1

        if permute:
            img = img.permute(2, 0, 1)
            mask = mask.permute(2, 0, 1)

        return img, mask

    def load_mask(self, mask_path: str, permute: bool = True):
        mask = np.array(self.random_crop_resize(Image.open(mask_path).convert("L")))
        mask = torch.from_numpy(mask / 255.0).unsqueeze(-1)

        if permute:
            mask = mask.permute(2, 0, 1)

        return mask
    
    def random_resize_mask(self, mask, min_scale, max_scale):

        channel, height, width = mask.shape
        scale_factor = random.uniform(min_scale, max_scale)
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        # interpolate mask
        scaled_mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(new_height, new_width), mode='nearest')
        scaled_mask = scaled_mask.squeeze(0)
        
        # initialize top and left
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        
        # create final mask
        final_mask = torch.zeros((channel, height, width), dtype=mask.dtype)
        
        if new_height <= height and new_width <= width:
            # if the scaled mask is smaller than the original mask, paste it in the center
            final_mask[:, top:top+new_height, left:left+new_width] = scaled_mask
        else:
            # if the scaled mask is larger than the original mask, crop it
            top = -top
            left = -left
            final_mask = scaled_mask[:, top:top+height, left:left+width]
        
        return final_mask

    def random_shift_mask(self, mask, max_shift):
        channel, height, width = mask.shape
        shifted_mask = torch.zeros_like(mask)
        tmp_shifted_mask = torch.zeros_like(mask)
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        
        if shift_x >= 0:
            tmp_shifted_mask[:, :, shift_x:] = mask[:, :, :-shift_x] if shift_x != 0 else mask
        else:
            tmp_shifted_mask[:, :, :shift_x] = mask[:, :, -shift_x:] if shift_x != 0 else mask

        if shift_y >= 0:
            shifted_mask[:, shift_y:, :] = tmp_shifted_mask[:, :-shift_y, :] if shift_y != 0 else mask
        else:
            shifted_mask[:, :shift_y, :] = tmp_shifted_mask[:, -shift_y:, :] if shift_y != 0 else mask

        return shifted_mask

    def __getitem__(self, index) -> Any:
        sample = self.data[index]
        target_image, target_mask = self.load_image(osp(self.root_dir, sample["target"]))

        # Load inpaint mask
        if random.random() < 0.5:
            inpaint_mask = make_mask(target_image, resolution=self.image_wh[0])
        else:
            inpaint_mask_path = random.choice(self.inpaint_mask_files)
            inpaint_mask = self.load_mask(inpaint_mask_path)
            inpaint_mask = self.random_resize_mask(inpaint_mask, 0.5, 1.2)
            inpaint_mask = self.random_shift_mask(inpaint_mask, self.image_wh[0] / 4)

        appearance, structure, mask = {}, {}, {}

        for key, value in sample["appearance"].items():
            ref_img, ref_mask = self.load_image(
                osp(self.root_dir, value), bg_color=self.bg_color
            )
            appearance[key] = ref_img
            mask[key] = ref_mask

        if not self.load_rgba and "mask" in sample:
            for key, value in sample["mask"].items():
                mask[key] = self.load_mask(osp(self.root_dir, value))

        return {
            "pixel_values": target_image,
            "target_mask": target_mask,
            "appearance": appearance,
            "structure": structure,
            "mask": mask,
            "caption": sample["caption"],
            # new inputs
            "inpaint_image": target_image,
            "inpaint_mask": inpaint_mask,
            "masked_image": target_image * (1 - inpaint_mask),
        }


if __name__ == "__main__":
    import torchvision
    from torch.utils.data import DataLoader

    dataset = RefTrgDataset(
        root_dir="data/DeepFashion-MultiModal-Parts2Whole",
        jsonl_path="data/DeepFashion-MultiModal-Parts2Whole/train.jsonl",
        image_wh=(768, 768),
        load_rgba=False,
        bg_color="gray",
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)

    print(len(dataset))
    for step, batch in enumerate(dataloader):
        print(step, batch["pixel_values"].shape)
        torchvision.utils.save_image(
            batch["pixel_values"] / 2 + 0.5, "check_target.jpg"
        )
        torchvision.utils.save_image(
            batch["target_mask"], "check_target_mask.jpg"
        )
        torchvision.utils.save_image(
            batch["inpaint_image"] / 2 + 0.5, "check_inpaint_image.jpg"
        )
        for i in range(8):
            torchvision.utils.save_image(
                batch["masked_image"][i] / 2 + 0.5, f"check_masked_image_{i}.jpg"
            )
        for i in range(8):
            torchvision.utils.save_image(batch["inpaint_mask"][i], f"check_inpaint_mask_{i}.jpg")
        for key, value in batch["appearance"].items():
            torchvision.utils.save_image(value / 2 + 0.5, f"check_{key}.jpg")
        for key, value in batch["structure"].items():
            torchvision.utils.save_image(value, f"check_{key}.jpg")
        for key, value in batch["mask"].items():
            torchvision.utils.save_image(value, f"check_{key}_mask.jpg")

        break
