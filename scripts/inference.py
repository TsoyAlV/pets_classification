import os
import sys
import json
import random
import time
from collections import Counter
from pathlib import Path

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
MODEL_NAME = config.get('MODEL_NAME', 'custom')
BATCH_SIZE = config.get('BATCH_SIZE', 64)
SAVE_MODEL_FILEDIR = config.get('SAVE_MODEL_FILEDIR', 64)
ANNOTATION_FILE_PATH = config.get('ANNOTATION_FILE_PATH', 'path_to_annotation')
ANNOTATION_FILE_PATH_TST = os.path.join(os.getcwd(), ANNOTATION_FILE_PATH, 'test.txt')


def main():
    d_test = DigitDataset('data/images', train=False, transform=None, annotation_file_path=ANNOTATION_FILE_PATH_TST)
    test_data = data.DataLoader(d_test, batch_size=BATCH_SIZE, shuffle=True)
    # выбор модели
    model = create_alexnet if MODEL_NAME == 'alexnet' else create_vgg if MODEL_NAME == 'vgg' else Net
    model = model()
    model.load_state_dict(torch.load(SAVE_MODEL_FILEDIR)['model_state_dict'])
    model.eval()
    acc, all_preds, all_labels = evaluate_model(model, test_data)

if __name__ == '__main__':
    main()
    
