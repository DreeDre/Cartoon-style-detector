import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CartoonDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = [d for d in os.listdir(root_dir)]
        self.class_to_imgs = {}
        self.imgs = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for cls in self.classes:
            img_dir = os.path.join(root_dir, cls)
            imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                    if f.endswith('.jpg')]
            self.class_to_imgs[cls] = imgs
            self.imgs.extend([(img_path, cls) for img_path in imgs])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        anchor_path, anchor_cls = self.imgs[idx]
        positive_path = anchor_path
        positives = [img for img in self.class_to_imgs[anchor_cls] if img != anchor_path]
        if len(positives) > 0:
            positive_path = random.choice(positives)
        negative_cls = random.choice([c for c in self.classes if c != anchor_cls])
        negative_path = random.choice(self.class_to_imgs[negative_cls])

        anchor = Image.open(anchor_path).convert('RGB')
        positive = Image.open(positive_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        anchor = self.transform(anchor)
        positive = self.transform(positive)
        negative = self.transform(negative)

        return anchor, positive, negative
