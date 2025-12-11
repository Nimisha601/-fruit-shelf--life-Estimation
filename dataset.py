
import os, pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class FruitShelfDataset(Dataset):
    def __init__(self, metadata_csv, img_root, transform=None):
        self.meta = pd.read_csv(metadata_csv).reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform
        labels = sorted(self.meta['class_label_text'].unique())
        self.class_to_idx = {c:i for i,c in enumerate(labels)}
        self.classes = labels

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = os.path.join(self.img_root, row['image_path'])
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            img = Image.new('RGB', (224, 224))
            
        if self.transform:
            img = self.transform(img)
            
        label = self.class_to_idx[row['class_label_text']]
        days = float(row['days_since_harvest'])
        meta = [float(row.get('storage_temp',0.0)), float(row.get('humidity',0.0))]
        return {'image': img, 'class': label, 'days': days, 'meta': meta}

def get_transforms(image_size=224):
    train = T.Compose([T.Resize((image_size,image_size)), T.RandomHorizontalFlip(), 
                       T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    val = T.Compose([T.Resize((image_size,image_size)), T.ToTensor(),
                     T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return train, val
