import os
import math
import numpy as np
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

# -------------------
# Гиперпараметры
# -------------------
IMG_SIZE = 112
BATCH_SIZE = 128
EPOCHS_FREEZE = 190
EPOCHS_FINE_TUNE = 50
LEARNING_RATE = 1e-4
LEARNING_RATE_FINE_TUNE = 1e-5

# Для фокальной функции потерь
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Нормировка (ImageNet)
IMGNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMGNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Проверка доступности GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# Предобработка изображения
# -------------------
def preprocess_image(image_path):
    """
    Открываем изображение, усиливаем резкость, 
    приводим к нужному размеру и нормируем по ImageNet-стандарту.
    Возвращаем NumPy-массив.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - IMGNET_MEAN) / IMGNET_STD  # нормировка
    return img  # (H, W, C) в формате NumPy

# -------------------
# Функция аугментации
# -------------------
# Используем torchvision transforms. 
# Аналогично ImageDataGenerator, делаем случайные повороты, флипы и т.д.
augmentation_transform = T.Compose([
    T.ToPILImage(),
    T.RandomRotation(45),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    T.ColorJitter(brightness=(0.8, 1.2)),
    T.ToTensor()
])

def random_augmentation(image_np):
    """
    Принимаем уже нормированный NumPy-массив (H,W,C).
    Применяем аугментацию и возвращаем NumPy-массив (H,W,C).
    """
    # Преобразуем из (H,W,C) в тензор PyTorch (C,H,W), применяем aug, возвращаем в (H,W,C)
    tensor_img = torch.from_numpy(image_np.transpose(2, 0, 1))  # (C,H,W)
    aug_img = augmentation_transform(tensor_img)  # (C,H,W)
    aug_img = aug_img.numpy().transpose(1, 2, 0)  # обратно в (H,W,C)
    # Перенормируем назад (aug сбросит наши mean/std), поэтому заново нормируем ниже:
    aug_img = (aug_img - IMGNET_MEAN) / IMGNET_STD
    return aug_img.astype(np.float32)

# -------------------
# Фокальная функция потерь для бинарной классификации
# -------------------
def focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    """
    FL = - alpha * (1 - p_t)^gamma * y * log(p_t)
         - (1-alpha) * (p_t)^gamma * (1 - y) * log(1 - p_t)
    """
    def loss_fn(preds, targets):
        # preds: тензор (N,), targets: тензор (N,)
        eps = 1e-7
        preds = torch.clamp(preds, eps, 1.0 - eps)
        
        ce = - (targets * torch.log(preds) + (1.0 - targets)*torch.log(1.0 - preds))
        p_t = targets * preds + (1.0 - targets)*(1.0 - preds)
        modulating_factor = (1.0 - p_t) ** gamma
        alpha_weight = targets * alpha + (1.0 - targets)*(1.0 - alpha)
        
        focal_ce = alpha_weight * modulating_factor * ce
        return focal_ce.mean()
    
    return loss_fn

# -------------------
# Создание PyTorch-модели на основе ResNet50
# -------------------
def build_model():
    """
    Создаём ResNet50 (pretrained=True) и меняем выходной слой
    на классификацию из 1 нейрона (бинарная классификация).
    """
    model = torchvision.models.resnet50(pretrained=True)
    # Заморозим сначала все параметры (на первом этапе)
    for param in model.parameters():
        param.requires_grad = False

    # Размер выхода у ResNet50 перед FC: 2048
    num_features = model.fc.in_features
    # Заменяем выходную часть (fc)
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid()  # т.к. бинарная классификация
    )
    return model

# -------------------
# Сборка датасета (из директорий)
# -------------------
def load_dataset(positive_dir, negative_dir):
    pos_files = [os.path.join(positive_dir, f) for f in os.listdir(positive_dir)]
    neg_files = [os.path.join(negative_dir, f) for f in os.listdir(negative_dir)]

    pos_data = [preprocess_image(f) for f in pos_files]
    neg_data = [preprocess_image(f) for f in neg_files]

    # По описанию создаём аугментированные версии
    k_aug = 2
    pos_aug_data = []
    for img in pos_data:
        for _ in range(k_aug):
            pos_aug_data.append(random_augmentation(img))

    neg_aug_data = []
    for img in neg_data:
        for _ in range(k_aug):
            neg_aug_data.append(random_augmentation(img))

    X = np.concatenate([pos_data, pos_aug_data, neg_data, neg_aug_data], axis=0)
    y = np.array(
        [1]* (len(pos_data)+len(pos_aug_data)) + 
        [0]* (len(neg_data)+len(neg_aug_data))
    )
    return X, y

# -------------------
# Класс Dataset для PyTorch
# -------------------
class CustomImageDataset(Dataset):
    def __init__(self, X, y):
        # X: NumPy массив (N, H, W, C), y: (N,)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Возвращаем тензор (C,H,W) и метку
        img_np = self.X[idx]
        label = self.y[idx]
        # Переводим img в тензор PyTorch
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))  # (C,H,W)
        return img_tensor, torch.tensor(label, dtype=torch.float32)

# -------------------
# Функция обучения (двухэтапная)
# -------------------
def train_model(X, y):
    """
    Разделяем X, y на train/val, 
    затем сначала обучаем «голову» на EPOCHS_FREEZE,
    потом размораживаем ResNet и обучаем EPOCHS_FINE_TUNE.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # Создаём датасеты и загрузчики
    train_ds = CustomImageDataset(X_train, y_train)
    val_ds = CustomImageDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model().to(device)
    criterion = focal_loss(FOCAL_ALPHA, FOCAL_GAMMA)
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # -------------------
    # 1) Обучение классификатора (замороженная ResNet50)
    # -------------------
    print("Шаг 1: Обучение классификатора на замороженной ResNet50...")
    model.train()
    for epoch in range(EPOCHS_FREEZE):
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze(1)  # (N,)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Валидация на эпохе
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).squeeze(1)
                val_loss += criterion(outputs, labels).item()
        model.train()

        print(f"Epoch [{epoch+1}/{EPOCHS_FREEZE}], "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

    # -------------------
    # 2) Размораживаем ResNet50 и обучаем всю модель
    # -------------------
    print("Шаг 2: Тонкая настройка всей модели (размороженная ResNet50)...")
    for param in model.parameters():
        param.requires_grad = True  # размораживаем все слои

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_FINE_TUNE)
    model.train()

    for epoch in range(EPOCHS_FINE_TUNE):
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Валидация
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).squeeze(1)
                val_loss += criterion(outputs, labels).item()
        model.train()

        print(f"Epoch [{epoch+1}/{EPOCHS_FINE_TUNE}], "
              f"Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

    return model

# -------------------
# Основной скрипт
# -------------------
if __name__ == "__main__":
    positive_dir = "dataset/positive_samples"
    negative_dir = "dataset/negative_samples"

    X, y = load_dataset(positive_dir, negative_dir)
    print(f"Всего данных: {X.shape}, Позитивных={sum(y)}, Негативных={len(y)-sum(y)}")

    trained_model = train_model(X, y)

    # Сохранение обученной модели
    torch.save(trained_model.state_dict(), "nazca_model.pth")
    print("Модель успешно сохранена в nazca_model.pth")