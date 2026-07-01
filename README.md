# Установка зависимостей

`pip install -r requirements.txt`

# Разработка моделей
Разработка моделей производилась в тетрадке "./notebooks/1. Pets classification.ipynb"

# Команды для обучения и инференса моделей
Находясь в корневой директории, можно выполнить следующие команды:
```
1) Обучение / Transfer learning:
  python scripts/train.py --epoches 5 
2) Инференс 
  python scripts/inference.py 
```
Веса модели и ее скор сохраняется в файле "data/best_mode.pth"

# config файл
```yaml [./configs/config.yaml]
NUM_CLASSES: 37 # число классов - выбирается из числа имеющихся классов в датасете (их 37)
MODEL_NAME: custom # vgg/alexnet/custom можно выбрать другую модель, но ее нужно добавить в файл ./scripts/models.py
NUM_EPOCHS: 1 # число эпох - для обучения достаточно 25-100 в зависимости от модели
LR: 0.001 # learning rate 
BATCH_SIZE: 64 # число картинок в батче
SAVE_MODEL_FILEDIR: 'data/best_model.pth' # директория по сохранению модели
ANNOTATION_FILE_PATH: 'data/annotations/' # файл с таргетами (trainval.txt, test.txt)
```

По идее можно было дообучить модели на accuracy больше 90-95%, но обучение было долгим. Оставил пока так.