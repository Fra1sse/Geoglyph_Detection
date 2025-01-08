import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

# Установим константы
IMG_SIZE = 112  # Размер изображения 112x112 пикселей
STEP_SIZE = 50  # Шаг для разбиения изображений
POSITIVE_RATIO = 150  # Количество положительных примеров
NEGATIVE_RATIO = 250  # Количество отрицательных примеров
LEARNING_RATE = 1e-4
EPOCHS_FREEZE = 190
EPOCHS_FINE_TUNE = 50
FOCAL_ALPHA = 0.25  # Вес классов
FOCAL_GAMMA = 2.0  # Параметр фокусировки

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

# === Этап 1. Подготовка данных === #

def preprocess_image(image_path):
    """Обработка изображения: усиление резкости и нормализация."""
    img = Image.open(image_path).filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Изменение размера
    img = np.array(img) / 255.0  # Scale to [0, 1]
    img = (img - IMGNET_MEAN) / IMGNET_STD
    return img

def augment_image(image):
    """Аугментация данных: случайные повороты, отражения и цветовые искажения."""
    datagen = ImageDataGenerator(
        rotation_range=45,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=(0.8, 1.2),
        zoom_range=0.2
    )
    return next(datagen.flow(np.expand_dims(image, axis=0), batch_size=128))[0]

# Загрузка данных
def load_dataset(positive_dir, negative_dir):
    """Загрузка и обработка изображений."""
    positive_images = [preprocess_image(os.path.join(positive_dir, f)) for f in os.listdir(positive_dir)]
    negative_images = [preprocess_image(os.path.join(negative_dir, f)) for f in os.listdir(negative_dir)]

    # Аугментация данных
    positive_aug = [augment_image(img) for img in positive_images]
    negative_aug = [augment_image(img) for img in negative_images]

    # Создание датасета
    X = np.array(positive_images + positive_aug + negative_images + negative_aug)
    y = np.array([1] * len(positive_images + positive_aug) + [0] * len(negative_images + negative_aug))
    return X, y

# === Этап 2. Определение модели === #

def focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss)
    return focal_loss_fixed

def build_model():
    """Создание модели на базе ResNet50."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False  # Заморозим веса

    # Добавляем классификатор
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# === Этап 3. Обучение модели === #

def train_model(X, y):
    """Обучение модели."""
    # Разделение на тренировочные и валидационные данные
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Шаг 1: Обучение классификатора
    model = build_model()
    model.compile(optimizer=AdamW(learning_rate=LEARNING_RATE),
                  loss=focal_loss(FOCAL_ALPHA, FOCAL_GAMMA),
                  metrics=['accuracy'])
    print("Шаг 1: Обучение классификатора на замороженной ResNet50...")
    model.fit(X_train, y_train, epochs=EPOCHS_FREEZE, validation_data=(X_val, y_val))

    # Шаг 2: Разморозка ResNet50 и дообучение
    print("Шаг 2: Разморозка ResNet50 и обучение всей модели...")
    model.layers[0].trainable = True  # Разморозить ResNet50
    model.compile(optimizer=AdamW(learning_rate=LEARNING_RATE / 10),
                  loss=focal_loss(FOCAL_ALPHA, FOCAL_GAMMA),
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS_FINE_TUNE, validation_data=(X_val, y_val))

    return model


# Пример использования
if __name__ == "__main__":
    positive_dir = "dataset/positive_samples"
    negative_dir = "dataset/negative_samples"
    X, y = load_dataset(positive_dir, negative_dir)
    model = train_model(X, y)

    model.save("nazca_model_2.h5")