import os
import sys
import json
import random
import time
from collections import Counter
from pathlib import Path
import click

import yaml
import numpy as np
import pandas as pd
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
from PIL import Image
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torchvision import transforms
import kagglehub

# Добавляем корневую папку проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.models import Net, create_alexnet, create_vgg, train_model, evaluate_model
from scripts.data import DigitDataset

"""Загрузка конфигурации из YAML файла"""
with open(os.path.join(os.path.dirname(__file__), '../configs/config.yaml'), 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
    
# constants
NUM_CLASSES = config.get('NUM_CLASSES', 37)
MODEL_NAME = config.get('MODEL_NAME', 'custom')
NUM_EPOCHS = config.get('NUM_EPOCHS', 10)
LR = config.get('LR', 1e-3)
BATCH_SIZE = config.get('BATCH_SIZE', 64)
SAVE_MODEL_FILEDIR = config.get('SAVE_MODEL_FILEDIR', 'path_to_model')
ANNOTATION_FILE_PATH = config.get('ANNOTATION_FILE_PATH', 'path_to_annotation')
ANNOTATION_FILE_PATH_TRN = os.path.join(os.getcwd(), ANNOTATION_FILE_PATH, 'trainval.txt')
ANNOTATION_FILE_PATH_TST = os.path.join(os.getcwd(), ANNOTATION_FILE_PATH, 'test.txt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

@click.command()
@click.option("--epoches", default=1, help="num of epoches")
def main(epoches):
    # формирование данных для обучения
    _tr = transforms.Compose([
        transforms.Resize((256, 256)),           # Сначала чуть больше
        transforms.RandomCrop(224),               # Random crop до 224x224
        transforms.RandomHorizontalFlip(p=0.5),   # Отражение по горизонтали
        transforms.RandomRotation(degrees=15),    # Поворот ±15°
        transforms.ColorJitter(
            brightness=0.2,    # Изменение яркости ±20%
            contrast=0.2,      # Контраст ±20%
            saturation=0.2,    # Насыщенность ±20%
            hue=0.1            # Оттенок ±10%
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Сдвиг до 10%
            scale=(0.9, 1.1)        # Масштаб 90%-110%
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    d_train = DigitDataset('data/images', transform=_tr, annotation_file_path=ANNOTATION_FILE_PATH_TRN)
    d_test = DigitDataset('data/images', train=False, transform=_tr, annotation_file_path=ANNOTATION_FILE_PATH_TST)
    train_data = data.DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True)
    test_data = data.DataLoader(d_test, batch_size=BATCH_SIZE, shuffle=True)
    # выбор модели
    model = create_alexnet if MODEL_NAME == 'alexnet' else create_vgg if MODEL_NAME == 'vgg' else Net
    model = model(NUM_CLASSES)
    model = model.to(DEVICE)
    history = train_model(
        model, 
        train_data, 
        test_data, 
        num_epochs=NUM_EPOCHS,
        lr=LR,
        device=DEVICE,
        save_filedir=SAVE_MODEL_FILEDIR
    )
    # inference
    acc, all_preds, all_labels = evaluate_model(model, test_data)


if __name__ == '__main__':
    path = kagglehub.dataset_download("julinmaloof/the-oxfordiiit-pet-dataset", output_dir='data')
    main()
    
