import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt


IMG_SIZE = 112  # Размер изображения 112x112 пикселей
STEP_SIZE = 50  # Шаг для разбиения изображений
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

# Преобразования для изображений
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD)
])

def preprocess_test_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
    img = np.array(img) / 255.0  # Scale to [0, 1]
    img = (img - IMGNET_MEAN) / IMGNET_STD
    return img

def split_image(image_path, step_size):
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size
    patches = []
    patch_coords = []  # Для хранения координат фрагментов
    for y in range(0, img_height - IMG_SIZE + 1, step_size):
        for x in range(0, img_width - IMG_SIZE + 1, step_size):
            patch = image.crop((x, y, x + IMG_SIZE, y + IMG_SIZE))
            patch = np.array(patch) / 255.0  # Нормализация
            patch = (patch - IMGNET_MEAN) / IMGNET_STD
            patch = torch.tensor(patch).permute(2, 0, 1).float()  
            patches.append(patch)
            patch_coords.append((x, y, x + IMG_SIZE, y + IMG_SIZE)) 
    return torch.stack(patches), patch_coords, image

# Загрузка модели
def load_model(model_path):
    model = resnet50(pretrained=False)
    num_features = model.fc.in_features
    # Воспроизводим структуру модели
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid()  
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Предсказание
def predict_geoglyphs(model, test_images, device):
    """Предсказание геоглифов на новых изображениях."""
    model.to(device)
    test_images = test_images.to(device)
    with torch.no_grad():
        predictions = model(test_images).squeeze(1)  
    candidates = [i for i, pred in enumerate(predictions.cpu().numpy()) if pred > 0.55]
    return candidates

# Визуализация результатов
def visualize_results(image, patch_coords, candidates):
    """Визуализация найденных фрагментов на исходном изображении."""
    draw = ImageDraw.Draw(image)
    for idx in candidates:
        x1, y1, x2, y2 = patch_coords[idx]
        # Рисуем прямоугольник вокруг найденного фрагмента
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Использование
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("nazca_model.pth")
    
    test_image_path = "test_1.jpg"
    test_patches, patch_coords, original_image = split_image(test_image_path, STEP_SIZE)
    candidates = predict_geoglyphs(model, test_patches, device)
    print("Кандидаты:", candidates)

    # Визуализация результатов
    visualize_results(original_image, patch_coords, candidates)