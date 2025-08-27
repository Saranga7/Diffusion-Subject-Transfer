from torch.utils.data import Dataset
from pathlib import Path
import random
import os
from PIL import Image
from torchvision import transforms


class DreamBoothDataset(Dataset):
    def __init__(self, 
                 instance_folder, 
                 class_folder, 
                 image_size = 512, 
                 instance_prompt_template="photo of {}",
                 class_prompt_template="photo of a {}", transforms=None):
        
        self.instance_paths = sorted([p for p in Path(instance_folder).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')])
        self.class_paths = sorted([p for p in Path(class_folder).iterdir() if p.suffix.lower() in ('.png','.jpg','.jpeg')])
       
        if len(self.instance_paths) == 0:
            raise ValueError(f"No instance images in {instance_folder}")
        if len(self.class_paths) == 0:
            raise ValueError(f"No class images in {class_folder}")
       
        self.image_size = image_size
        self.transforms = transforms or transforms_default(image_size)
        self.instance_prompt_template = instance_prompt_template
        self.class_prompt_template = class_prompt_template
        

    def __len__(self):
        return max(len(self.instance_paths), len(self.class_paths))

    def __getitem__(self, idx):
        inst_p = self.instance_paths[idx % len(self.instance_paths)]
        cls_p = self.class_paths[random.randint(0, len(self.class_paths)-1)]
        inst = Image.open(inst_p).convert("RGB")
        cls = Image.open(cls_p).convert("RGB")
        return self.transforms(inst), self.transforms(cls)

def transforms_default(image_size):
    return transforms.Compose([
        transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


