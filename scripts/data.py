from PIL import Image
import torch
import torch.utils.data as data
from torchvision.transforms import v2
from torchvision import transforms
from pathlib import Path

class DigitDataset(data.Dataset):
    def __init__(self, path, train=True, annotation_file_path=None, transform=None):
        if not annotation_file_path:
            if train:
                annotation_file_path = Path.joinpath(Path(path), '../annotations', 'trainval.txt')
            else:
                annotation_file_path = Path.joinpath(Path(path), '../annotations', 'test.txt')
        else:
            annotation_file_path = Path(annotation_file_path)
        self.path = path
        self.length = 0
        self.targets = torch.eye(37)
        # Стандартные трансформации, если не переданы
        # Если transform не передан, создаем стандартный
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Приводим все изображения к 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.2], std=[0.5])
            ])
        else:
            self.transform = transform
        with open(annotation_file_path, 'r') as f:
            labels = f.read()
            labels = labels.split('\n')[6:][:-1]
            labels = dict(zip([self.path + '/' + i.split()[0]+'.jpg' for i in labels], [int(i.split()[1]) for i in labels]))
            
        data = labels.items()
        self.length = len(data)
        self.files = list(data)
            
    def __getitem__(self, item):
        path_file, target = self.files[item]
        img = Image.open(path_file).convert('RGB')#.convert('L' if self.transform else 'RGB')  # Для черно-белых изображений
        if self.transform:
            img = self.transform(img)
        t = self.targets[target-1]
        return img, t
 
    def __len__(self):
        return self.length
