import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dataset_path = "/kaggle/input/productsdefects-nodefects-01-03-25/productsDefects-noDefects-01-03-25"
train_images_path = os.path.join(dataset_path, "images/train")
train_masks_path = os.path.join(dataset_path, "annotations/train")

dataset_path_test = "/kaggle/input/productsdefects-nodefects-01-03-25/productsDefects-noDefects-01-03-25"
test_images_path = os.path.join(dataset_path_test, "images/test")
test_masks_path = os.path.join(dataset_path_test, "annotations/test")

print("Train Images:", len(os.listdir(train_images_path)))
print("Train Masks:", len(os.listdir(train_masks_path)))
print("Test Images:", len(os.listdir(test_images_path)))
print("Test Masks:", len(os.listdir(test_masks_path)))

print("\nПримеры файлов (Train Images):", os.listdir(train_images_path)[:5])
print("Примеры файлов (Train Masks):", os.listdir(train_masks_path)[:5])

# Функция для загрузки и отображения изображения и маски
def visualize_sample(image_path, mask_path):
    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        return
    if not os.path.exists(mask_path):
        print(f"Маска не найдена: {mask_path}")
        return

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)

    print(f"Изображение: {image.shape}, Маска: {mask.shape}, dtype: {mask.dtype}")
    print(f"Уникальные значения в маске: {np.unique(mask)}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Исходное изображение")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Маска (0 = фон, 255 = дефект)")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()

image_files = sorted(os.listdir(train_images_path))
sample_image = image_files[0]
sample_mask = os.path.splitext(sample_image)[0] + ".png"
image_path = os.path.join(train_images_path, sample_image)
mask_path = os.path.join(train_masks_path, sample_mask)
visualize_sample(image_path, mask_path)

#dataset
class Imitation_SWIR_dataset(Dataset):
    """
    Dataset для сегментации дефектов.
    - Входное изображение: grayscale, копируется в 3 канала
    - Выходная маска: бинарная [1, H, W] с значениями [0, 1]
    """
    def __init__(self, images_dir, masks_dir, transform=None, img_size=512):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_filenames = sorted(os.listdir(images_dir))
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        mask_filename = os.path.splitext(image_filename)[0] + ".png"
        image_path = os.path.join(self.images_dir, image_filename)
        mask_path = os.path.join(self.masks_dir, mask_filename)

        #ЗАГРУЗКА ИЗОБРАЖЕНИЯ
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        image = cv2.resize(image, (self.img_size, self.img_size))  # [H, W]
        image = image.astype(np.float32) / 255.0  # Нормализация [0, 1]

        # Копируем grayscale в 3 канала (имитация IR)
        image = np.stack([image, image, image], axis=-1)  # [H, W, 3]

        #ЗАГРУЗКА МАСКИ
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask, dtype=np.float32)  # УЖЕ [0, 1], НЕ ДЕЛАЕМ / 255.0

        # Если маска содержит [0, 1], оставляем как есть
        # Если вдруг будут [0, 255], раскомментируйте:
        # if mask.max() > 1:
        #     mask = mask / 255.0

        # AUGMENTATION (опционально)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # TENSOR CONVERSION
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # [3, H, W]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return image, mask

# ТЕСТИРОВАНИЕ DATASET

print("ТЕСТИРОВАНИЕ DATASET")

train_dataset = Imitation_SWIR_dataset(train_images_path, train_masks_path, img_size=512)

print(f"\nDataset создан успешно")
print(f"Размер датасета: {len(train_dataset)} сэмплов\n")

# Загружаем один сэмпл
sample_img, sample_mask = train_dataset[0]

print("--- Проверка одного сэмпла ---")
print(f"Форма изображения: {sample_img.shape}")  # Ожидаем [3, 512, 512]
print(f"Тип данных: {sample_img.dtype}")
print(f"Диапазон: [{sample_img.min():.4f}, {sample_img.max():.4f}]")

print(f"Форма маски: {sample_mask.shape}")  # Ожидаем [1, 512, 512]
print(f"ип данных: {sample_mask.dtype}")
print(f"Диапазон: [{sample_mask.min():.4f}, {sample_mask.max():.4f}]")
print(f"Уникальные значения: {torch.unique(sample_mask).numpy()}")

# Проверяем, нет ли NaN
print(f"Проверка на NaN/Inf")
print(f"Image содержит NaN: {torch.isnan(sample_img).any().item()}")
print(f"Image содержит Inf: {torch.isinf(sample_img).any().item()}")
print(f"Mask содержит NaN: {torch.isnan(sample_mask).any().item()}")
print(f"Mask содержит Inf: {torch.isinf(sample_mask).any().item()}")

# Визуализируем сэмпл
print(f"Визуализация сэмпла")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Первый канал изображения
axes[0].imshow(sample_img[0].numpy(), cmap='gray')
axes[0].set_title("Изображение (Channel 0)")
axes[0].axis("off")

# Все 3 канала (они одинаковые)
axes[1].imshow(sample_img.permute(1, 2, 0).numpy()[:, :, 0], cmap='gray')
axes[1].set_title("Изображение (Визуализация)")
axes[1].axis("off")

# Маска
axes[2].imshow(sample_mask.squeeze().numpy(), cmap='gray')
axes[2].set_title("Маска сегментации")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# Загружаем несколько сэмплов для проверки
print(f"Загрузка 5 случайных сэмплов")
indices = np.random.choice(len(train_dataset), 5, replace=False)
for i, idx in enumerate(indices):
    img, mask = train_dataset[idx]
    print(f"Сэмпл {i+1}: Image {img.shape}, Mask {mask.shape}, "
          f"Mask values: {torch.unique(mask).numpy()}")


#  ТЕСТИРОВАНИЕ DATALOADER

print("ТЕСТИРОВАНИЕ DATALOADER")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

print(f"DataLoader создан успешно")
print(f"Количество батчей: {len(train_dataloader)}")
print(f"Размер батча:")

# Загружаем первый батч
batch_img, batch_mask = next(iter(train_dataloader))

print("--- Проверка одного батча ---")
print(f"Батч изображений: {batch_img.shape}")  # [8, 3, 512, 512]
print(f" Тип данных: {batch_img.dtype}")
print(f"Диапазон: [{batch_img.min():.4f}, {batch_img.max():.4f}]")

print(f"Батч масок: {batch_mask.shape}")  # [8, 1, 512, 512]
print(f"Тип данных: {batch_mask.dtype}")
print(f"Диапазон: [{batch_mask.min():.4f}, {batch_mask.max():.4f}]")

# Проверяем баланс классов в батче
print(f"Баланс классов в батче")
for i in range(batch_mask.shape[0]):
    mask = batch_mask[i].squeeze()
    num_defects = (mask > 0.5).sum().item()
    num_background = (mask <= 0.5).sum().item()
    ratio = num_defects / (num_defects + num_background) * 100 if (num_defects + num_background) > 0 else 0
    print(f"  Сэмпл {i}: Background={num_background}, Defects={num_defects}, "
          f"Ratio={ratio:.1f}%")

# Визуализируем батч
print(f"Визуализация батча (первые 4 сэмпла)")
fig, axes = plt.subplots(4, 3, figsize=(12, 12))

for i in range(4):
    # Изображение
    axes[i, 0].imshow(batch_img[i, 0].numpy(), cmap='gray')
    axes[i, 0].set_title(f"Image {i}")
    axes[i, 0].axis("off")

    # Маска
    axes[i, 1].imshow(batch_mask[i, 0].numpy(), cmap='gray')
    axes[i, 1].set_title(f"Mask {i}")
    axes[i, 1].axis("off")

    # Оверлей (маска на изображении)
    overlay = batch_img[i, 0].numpy().copy()
    mask_overlay = batch_mask[i, 0].numpy()
    axes[i, 2].imshow(overlay, cmap='gray', alpha=0.6)
    axes[i, 2].imshow(mask_overlay, cmap='Reds', alpha=0.4)
    axes[i, 2].set_title(f"Overlay {i}")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()

# Проверяем на утечки памяти (загружаем несколько батчей)
print(f"Проверка на утечки памяти (загружаем 10 батчей)")
for batch_idx, (batch_img, batch_mask) in enumerate(train_dataloader):
    if batch_idx >= 10:
        break
    print(f"Батч {batch_idx}: img {batch_img.shape}, mask {batch_mask.shape}, "
          f"GPU память (если CUDA): {torch.cuda.memory_allocated() / 1e6:.2f} MB")

import torch.nn as nn
import torchvision.models.segmentation as segmentation
from torch.optim import Adam, SGD
import torch.optim.lr_scheduler as lr_scheduler

#  ЗАГРУЗКА И АДАПТАЦИЯ МОДЕЛИ

# Загружаем pretrained модель
model = segmentation.deeplabv3_resnet50(pretrained=True)

# АДАПТАЦИЯ ДЛЯ БИНАРНОЙ СЕГМЕНТАЦИИ

# DeepLabV3 по умолчанию имеет 21 класс (COCO)
# Нам нужен 1 класс (дефект vs фон) для BCEWithLogitsLoss

# Заменяем последний слой classifier (главный выход)
# Вместо Conv2d(256, 21, kernel_size=1) → Conv2d(256, 1, kernel_size=1)
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

# Заменяем последний слой aux_classifier (вспомогательный выход)
model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=1)

print("Адаптированы для бинарной сегментации:")
print("   - classifier[4]: Conv2d(256, 1, kernel_size=1)")
print("   - aux_classifier[4]: Conv2d(256, 1, kernel_size=1)")

#  ПРОВЕРКА СТРУКТУРЫ

print("Проверка структуры classifier после адаптации")
print(model.classifier)
print("Проверка структуры aux_classifier после адаптации")
print(model.aux_classifier)

# УСТРОЙСТВО
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# ПОДСЧЕТ ПАРАМЕТРОВ

#ВСЕ параметры разморозены
for param in model.backbone.parameters():
    param.requires_grad = True

# 3. Разморозить heads
for param in model.classifier.parameters():
    param.requires_grad = True
for param in model.aux_classifier.parameters():
    param.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# ПОДСЧЕТ ПАРАМЕТРОВ

print(f"Параметры модели:")
print(f"Всего параметров: {total_params:,}")
print(f"Обучаемых параметров: {trainable_params:,}")

#LOSS FUNCTION

# Для бинарной сегментации используем BCEWithLogitsLoss с pos_weight
# pos_weight штрафует false negatives на дефектах (которых мало)

# Вычислим pos_weight на основе датасета
total_background_pixels = 0
total_defect_pixels = 0

# Проходим по датасету и считаем пиксели
for idx in range(len(train_dataset)):
    _, mask = train_dataset[idx]
    total_background_pixels += (mask <= 0.5).sum().item()
    total_defect_pixels += (mask > 0.5).sum().item()

pos_weight = total_background_pixels / max(total_defect_pixels, 1)
print(f"Всего пикселей фона: {total_background_pixels:,}")
print(f"Всего пикселей дефектов: {total_defect_pixels:,}")
print(f"Соотношение: {pos_weight:.2f}")

pos_weight = torch.tensor([pos_weight]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

print(f"Loss функция: BCEWithLogitsLoss")
print(f"pos_weight: {pos_weight.item():.2f}")

def compute_iou(predictions, targets, threshold=0.5):
    """
    Вычисляет IoU (Intersection over Union) для бинарной сегментации.

    Args:
        predictions: тензор [B, 1, H, W] с логитами
        targets: тензор [B, 1, H, W] с значениями [0, 1]
        threshold: порог для преобразования в бинарные предсказания

    Returns:
        mean_iou: среднее IoU по батчу
    """
    # Преобразуем логиты в вероятности
    probs = torch.sigmoid(predictions)

    # Применяем порог
    preds_binary = (probs > threshold).float()

    # Вычисляем пересечение и объединение
    intersection = (preds_binary * targets).sum(dim=(1, 2, 3))
    union = (preds_binary + targets).sum(dim=(1, 2, 3)) - intersection

    # Добавляем небольшое значение, чтобы избежать деления на ноль
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean().item()

def compute_dice(predictions, targets, threshold=0.5):
    """
    Вычисляет Dice coefficient (F1-score для сегментации).

    Args:
        predictions: тензор [B, 1, H, W] с логитами
        targets: тензор [B, 1, H, W] с значениями [0, 1]
        threshold: порог для преобразования в бинарные предсказания

    Returns:
        mean_dice: средний Dice по батчу
    """
    probs = torch.sigmoid(predictions)
    preds_binary = (probs > threshold).float()

    intersection = (preds_binary * targets).sum(dim=(1, 2, 3))
    pred_sum = preds_binary.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))

    # Если оба пусты (pred_sum = 0 и target_sum = 0) → Dice = 1
    dice = torch.where(
        (pred_sum == 0) & (target_sum == 0),
        torch.ones_like(intersection, dtype=torch.float32),  # Dice = 1
        (2.0 * intersection) / (pred_sum + target_sum + 1e-6)
    )

    return dice.mean().item()

# HISTORY
from tqdm import tqdm
from datetime import datetime
import json

# OPTIMIZER

# Этап 1: Обучаем всю сеть с нормальным LR

# SGD традиционно работает лучше для сегментации (используется в оригинальной DeepLabV3)
optimizer = SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4
)

# SCHEDULER

# Уменьшаем LR каждые N эпох
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

#ПАРАМЕТРЫ ОБУЧЕНИЯ

batch_size = 8
aux_loss_weight = 0.4  # Вес вспомогательного классификатора (выбран из оригинальной статьи)

os.makedirs('checkpoints', exist_ok=True)
CHECKPOINT_DIR = 'checkpoints'
NUM_EPOCHS = 30


history = {
    'epoch': [],
    'train_loss': [],
    'train_aux_loss': [],
    'train_iou': [],
    'train_dice': [],
    'learning_rate': []
}

best_iou = 0.0
best_epoch = 0

# ОБУЧЕНИЕ

start_time = datetime.now()

for epoch in range(NUM_EPOCHS):
    model.train()

    epoch_loss = 0.0
    epoch_aux_loss = 0.0
    epoch_iou = 0.0
    epoch_dice = 0.0
    num_batches = 0

    # Progress bar
    pbar = tqdm(
        train_dataloader,
        desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]",
        total=len(train_dataloader),
        leave=True
    )

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # FORWARD PASS
        optimizer.zero_grad()
        outputs = model(images)

        main_output = outputs['out']  # [B, 1, H, W]
        aux_output = outputs['aux']   # [B, 1, H, W]

        # LOSS
        loss = criterion(main_output, masks)
        aux_loss = criterion(aux_output, masks)
        total_loss = loss + aux_loss_weight * aux_loss

        # BACKWARD PASS
        total_loss.backward()

        # Gradient clipping для стабильности
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        #  МЕТРИКИ
        iou = compute_iou(main_output, masks)
        dice = compute_dice(main_output, masks)

        epoch_loss += loss.item()
        epoch_aux_loss += aux_loss.item()
        epoch_iou += iou
        epoch_dice += dice
        num_batches += 1

        #  UPDATE PROGRESS BAR
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Aux Loss': f'{aux_loss.item():.4f}',
            'IoU': f'{iou:.4f}',
            'Dice': f'{dice:.4f}'
        })

    # УСРЕДНЯЕМ ЗА ЭПОХУ
    avg_loss = epoch_loss / num_batches
    avg_aux_loss = epoch_aux_loss / num_batches
    avg_iou = epoch_iou / num_batches
    avg_dice = epoch_dice / num_batches
    current_lr = optimizer.param_groups[0]['lr']

    #  SCHEDULER STEP
    scheduler.step()

    #  СОХРАНЕНИЕ В HISTORY
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(avg_loss)
    history['train_aux_loss'].append(avg_aux_loss)
    history['train_iou'].append(avg_iou)
    history['train_dice'].append(avg_dice)
    history['learning_rate'].append(current_lr)

    #  ЛОГИРОВАНИЕ
    print(f" Epoch [{epoch+1}/{NUM_EPOCHS}]")
    print(f"   Loss: {avg_loss:.6f} | Aux Loss: {avg_aux_loss:.6f}")
    print(f"   IoU: {avg_iou:.6f} | Dice: {avg_dice:.6f}")
    print(f"   Learning Rate: {current_lr:.8f}")

    #  СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ
    if avg_iou > best_iou:
        best_iou = avg_iou
        best_epoch = epoch + 1
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'Stage1-DLv3_RN50_Imitation_SWIR_Dataset_best.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f" Лучшая модель сохранена (Epoch {best_epoch}, IoU: {best_iou:.6f})")

#  ИТОГИ

end_time = datetime.now()
training_time = end_time - start_time

print(f"Время обучения: {training_time}")
print(f"Лучший IoU: {best_iou:.6f} (Epoch {best_epoch})")
print(f"Финальный IoU: {history['train_iou'][-1]:.6f}")
print(f"Финальный Loss: {history['train_loss'][-1]:.6f}")

# СОХРАНЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ

final_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'Stage1-DLv3_RN50_Imitation_SWIR_Dataset_final.pth')
torch.save(model.state_dict(), final_checkpoint_path)
print(f"Финальная модель сохранена: {final_checkpoint_path}")

# СОХРАНЕНИЕ ИСТОРИИ

history_path = os.path.join(CHECKPOINT_DIR, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history, f, indent=4)
print(f"История обучения сохранена: {history_path}")

#  ГРАФИКИ


fig = plt.figure(figsize=(16, 12))

#  ГРАФИК 1: Training Loss
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history['epoch'], history['train_loss'], 'b-', linewidth=2.5, label='Main Loss')
ax1.plot(history['epoch'], history['train_aux_loss'], 'r-', linewidth=2.5, label='Aux Loss')
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training Loss per Epoch', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, NUM_EPOCHS)

#  ГРАФИК 2: Mean IoU
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history['epoch'], history['train_iou'], 'g-', linewidth=2.5, marker='o', markersize=4)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean IoU', fontsize=12, fontweight='bold')
ax2.set_title('Mean IoU per Epoch', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, NUM_EPOCHS)
best_iou_idx = history['train_iou'].index(max(history['train_iou']))
ax2.plot(history['epoch'][best_iou_idx], history['train_iou'][best_iou_idx],
         'r*', markersize=20, label=f'Best: {max(history["train_iou"]):.6f}')
ax2.legend(fontsize=11)

#  ГРАФИК 3: Mean Dice
ax3 = plt.subplot(2, 3, 3)
ax3.plot(history['epoch'], history['train_dice'], 'purple', linewidth=2.5, marker='s', markersize=4)
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Mean Dice', fontsize=12, fontweight='bold')
ax3.set_title('Mean Dice per Epoch', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(1, NUM_EPOCHS)
best_dice_idx = history['train_dice'].index(max(history['train_dice']))
ax3.plot(history['epoch'][best_dice_idx], history['train_dice'][best_dice_idx],
         'r*', markersize=20, label=f'Best: {max(history["train_dice"]):.6f}')
ax3.legend(fontsize=11)

#  ГРАФИК 4: Learning Rate
ax4 = plt.subplot(2, 3, 4)
ax4.plot(history['epoch'], history['learning_rate'], 'orange', linewidth=2.5, marker='^', markersize=4)
ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax4.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, which='both')
ax4.set_yscale('log')
ax4.set_xlim(1, NUM_EPOCHS)

#  ГРАФИК 5: Loss Trend (логарифмическая шкала)
ax5 = plt.subplot(2, 3, 5)
ax5.semilogy(history['epoch'], history['train_loss'], 'b-', linewidth=2.5, label='Main Loss')
ax5.semilogy(history['epoch'], history['train_aux_loss'], 'r-', linewidth=2.5, label='Aux Loss')
ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax5.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
ax5.set_title('Training Loss (Log Scale)', fontsize=14, fontweight='bold')
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3, which='both')
ax5.set_xlim(1, NUM_EPOCHS)

# ГРАФИК 6: IoU vs Dice
ax6 = plt.subplot(2, 3, 6)
ax6.plot(history['epoch'], history['train_iou'], 'g-', linewidth=2.5, marker='o', markersize=4, label='IoU')
ax6.plot(history['epoch'], history['train_dice'], 'purple', linewidth=2.5, marker='s', markersize=4, label='Dice')
ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax6.set_ylabel('Score', fontsize=12, fontweight='bold')
ax6.set_title('IoU vs Dice Comparison', fontsize=14, fontweight='bold')
ax6.legend(fontsize=11)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(1, NUM_EPOCHS)

plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_plots_stage1.png'), dpi=300, bbox_inches='tight')
print(f"Графики сохранены: {os.path.join(CHECKPOINT_DIR, 'training_plots_stage1.png')}")
plt.show()

# ФИНАЛЬНЫЙ ОТЧЕТ


print(f" Статистика обучения:")
print(f"   Начальный IoU: {history['train_iou'][0]:.6f}")
print(f"   Финальный IoU: {history['train_iou'][-1]:.6f}")
print(f"   Лучший IoU: {max(history['train_iou']):.6f} (Epoch {best_epoch})")
print(f"   Улучшение: {(history['train_iou'][-1] - history['train_iou'][0]):.6f}")

print(f" Начальный Dice: {history['train_dice'][0]:.6f}")
print(f"   Финальный Dice: {history['train_dice'][-1]:.6f}")
print(f"   Лучший Dice: {max(history['train_dice']):.6f}")

print(f" Начальный Loss: {history['train_loss'][0]:.6f}")
print(f"   Финальный Loss: {history['train_loss'][-1]:.6f}")
print(f"   Снижение Loss: {(history['train_loss'][0] - history['train_loss'][-1]):.6f}")

print(f" Время обучения: {training_time}")
print(f" Лучшая модель: {os.path.join(CHECKPOINT_DIR, 'Stage1-DLv3_RN50_Imitation_SWIR_Dataset_best.pth')}")
print(f" Финальная модель: {final_checkpoint_path}")

import shutil
shutil.make_archive('/kaggle/working/checkpoints_dlv3-rn50-stage1', 'zip', '/kaggle/working/checkpoints')

test_dataset = Imitation_SWIR_dataset(test_images_path, test_masks_path, img_size=512)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

checkpoint_path = 'checkpoints/Stage1-DLv3_RN50_Imitation_SWIR_Dataset_best.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print(f"Модель загружена: {checkpoint_path}")

model.eval()

def compute_iou_with_empty_mask_handling(predictions, targets, threshold=0.5):
    """
    Вычисляет IoU с правильной обработкой пустых масок.

    Если обе маски пусты (нет дефектов) → IoU = 1.0
    """
    probs = torch.sigmoid(predictions)
    preds_binary = (probs > threshold).float()

    intersection = (preds_binary * targets).sum(dim=(1, 2, 3))
    union = (preds_binary + targets).sum(dim=(1, 2, 3)) - intersection

    #Если union = 0 → обе маски пусты → IoU = 1
    iou = torch.where(
        union == 0,
        torch.ones_like(union),
        (intersection + 1e-6) / (union + 1e-6)
    )

    return iou


def compute_dice_with_empty_mask_handling(predictions, targets, threshold=0.5):
    """
    Вычисляет Dice с правильной обработкой пустых масок.

    Если обе маски пусты (нет дефектов) → Dice = 1.0
    """
    probs = torch.sigmoid(predictions)
    preds_binary = (probs > threshold).float()

    intersection = (preds_binary * targets).sum(dim=(1, 2, 3))
    pred_sum = preds_binary.sum(dim=(1, 2, 3))
    target_sum = targets.sum(dim=(1, 2, 3))

    # Если оба пусты (pred_sum = 0 и target_sum = 0) → Dice = 1
    dice = torch.where(
        (pred_sum == 0) & (target_sum == 0),
        torch.ones_like(intersection, dtype=torch.float32),
        (2.0 * intersection) / (pred_sum + target_sum + 1e-6)
    )

    return dice

# Метрики для изображений С дефектами
iou_scores_with_defects = []
dice_scores_with_defects = []
indices_with_defects = []

# Метрики для изображений БЕЗ дефектов
false_positive_scores = []  # FP / (FP + TN) - для каждого изображения
indices_without_defects = []

all_predictions = []
all_targets = []

with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(tqdm(test_dataloader, desc="Testing", total=len(test_dataloader))):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        # DeepLabV3, берём только основной выход
        if isinstance(outputs, dict):
            main_output = outputs['out']
        else:
            main_output = outputs

        # Преобразуем в вероятности
        probs = torch.sigmoid(main_output)
        preds_binary = (probs > 0.5).float()

        # Проверяем, есть ли дефекты в ground truth
        target_sum = masks.sum(dim=(1, 2, 3))
        has_defects = target_sum > 0

        # Если есть дефекты → добавляем в список IoU/Dice
        if has_defects.item():
            # Вычисляем IoU и Dice
            iou_tensor = compute_iou_with_empty_mask_handling(main_output, masks)
            dice_tensor = compute_dice_with_empty_mask_handling(main_output, masks)

            iou_scores_with_defects.append(iou_tensor.item())
            dice_scores_with_defects.append(dice_tensor.item())
            indices_with_defects.append(batch_idx)
        else:
            # Если нет дефектов → вычисляем False Positive Rate
            # FP = предсказали дефект, но его нет
            fp = preds_binary.sum(dim=(1, 2, 3)).item()

            # TN = правильно предсказали, что дефекта нет
            # Всего пикселей минус FP
            total_pixels = masks.shape[2] * masks.shape[3]
            tn = total_pixels - fp

            # FP Rate = FP / (FP + TN)
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            false_positive_scores.append(fp_rate)
            indices_without_defects.append(batch_idx)

        # preds_binary уже определена для обоих случаев
        all_predictions.append(preds_binary.cpu().numpy())
        all_targets.append(masks.cpu().numpy())

# Статистика для изображений с дефектами
print(f"ИЗОБРАЖЕНИЯ С ДЕФЕКТАМИ: {len(iou_scores_with_defects)}")
if len(iou_scores_with_defects) > 0:
    mean_iou = np.mean(iou_scores_with_defects)
    std_iou = np.std(iou_scores_with_defects)
    min_iou = np.min(iou_scores_with_defects)
    max_iou = np.max(iou_scores_with_defects)

    mean_dice = np.mean(dice_scores_with_defects)
    std_dice = np.std(dice_scores_with_defects)
    min_dice = np.min(dice_scores_with_defects)
    max_dice = np.max(dice_scores_with_defects)

    print(f"IoU (Intersection over Union):")
    print(f"      Mean: {mean_iou:.6f}")
    print(f"      Std:  {std_iou:.6f}")
    print(f"      Min:  {min_iou:.6f}")
    print(f"      Max:  {max_iou:.6f}")

    print(f"Dice Coefficient:")
    print(f"      Mean: {mean_dice:.6f}")
    print(f"      Std:  {std_dice:.6f}")
    print(f"      Min:  {min_dice:.6f}")
    print(f"      Max:  {max_dice:.6f}")
else:
    print("Нет изображений с дефектами!")

# Статистика для изображений без дефектов
print(f"ИЗОБРАЖЕНИЯ БЕЗ ДЕФЕКТОВ: {len(false_positive_scores)}")
if len(false_positive_scores) > 0:
    fp_rate_mean = np.mean(false_positive_scores)
    fp_rate_std = np.std(false_positive_scores)
    fp_rate_min = np.min(false_positive_scores)
    fp_rate_max = np.max(false_positive_scores)

    print(f"False Positive Rate (по изображениям):")
    print(f"      Mean: {fp_rate_mean:.6f} ({fp_rate_mean*100:.2f}%)")
    print(f"      Std:  {fp_rate_std:.6f}")
    print(f"      Min:  {fp_rate_min:.6f} ({fp_rate_min*100:.2f}%)")
    print(f"      Max:  {fp_rate_max:.6f} ({fp_rate_max*100:.2f}%)")
else:
    print("   Нет изображений без дефектов!")

print(f"ОБЩАЯ СТАТИСТИКА:")
print(f"   Всего тестовых образцов:     {len(test_dataset)}")
print(f"   С дефектами:                 {len(iou_scores_with_defects)} ({len(iou_scores_with_defects)/len(test_dataset)*100:.1f}%)")
print(f"   Без дефектов:                {len(false_positive_scores)} ({len(false_positive_scores)/len(test_dataset)*100:.1f}%)")

# ВЫЧИСЛЕНИЕ mAP
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt


#  ПОДГОТОВКА ДАННЫХ

# Собираем все вероятности и targets
all_probs_flat = np.concatenate([p.flatten() for p in all_probs])  # Используем все пиксели
all_targets_flat = np.concatenate([t.flatten() for t in all_targets])

# mAP на уровне пикселей
mAP_pixel = average_precision_score(all_targets_flat, all_probs_flat)

print(f"mAP (pixel-level): {mAP_pixel:.6f}")

#  ВЫЧИСЛЕНИЕ mAP НА УРОВНЕ ИЗОБРАЖЕНИЙ

# Для каждого изображения считаем Average Precision
image_ap_scores = []

with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_dataloader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        if isinstance(outputs, dict):
            main_output = outputs['out']
        else:
            main_output = outputs

        # Преобразуем в вероятности
        probs = torch.sigmoid(main_output)

        # Для уровня изображений: макимальная вероятность в предсказанной маске
        # или средняя вероятность по маске
        image_pred_prob = probs.max().item()  # Максимальная вероятность
        image_target = masks.sum().item() > 0  # Есть ли дефект

        # Flatten для расчета AP по пикселям в каждом изображении
        probs_flat = probs.cpu().numpy().flatten()
        masks_flat = masks.cpu().numpy().flatten()

        # Average Precision для этого изображения
        if np.sum(masks_flat) > 0:  # Если есть дефекты
            ap = average_precision_score(masks_flat, probs_flat)
            image_ap_scores.append(ap)

# mAP на уровне изображений
if len(image_ap_scores) > 0:
    mAP_image = np.mean(image_ap_scores)
    print(f"mAP (image-level, только для изображений с дефектами):")
    print(f"   Mean mAP: {mAP_image:.6f}")
    print(f"   Std:      {np.std(image_ap_scores):.6f}")
    print(f"   Min:      {np.min(image_ap_scores):.6f}")
    print(f"   Max:      {np.max(image_ap_scores):.6f}")

# PRECISION-RECALL CURVE

precision, recall, thresholds = precision_recall_curve(all_targets_flat, all_probs_flat)

# Area Under Precision-Recall Curve (AUPRC)
auprc = average_precision_score(all_targets_flat, all_probs_flat)

print(f" AUPRC (Area Under Precision-Recall Curve): {auprc:.6f}")

# ГРАФИКИ

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Precision-Recall Curve
axes[0].plot(recall, precision, color='blue', linewidth=2.5, label=f'PR Curve (AP={auprc:.4f})')
axes[0].fill_between(recall, precision, alpha=0.2, color='blue')
axes[0].set_xlabel('Recall', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Precision', fontsize=12, fontweight='bold')
axes[0].set_title('Precision-Recall Curve (Pixel-level)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

# График 2: Распределение AP по изображениям
if len(image_ap_scores) > 0:
    axes[1].hist(image_ap_scores, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(mAP_image, color='red', linestyle='--', linewidth=2.5, label=f'Mean AP: {mAP_image:.4f}')
    axes[1].set_xlabel('Average Precision', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title(f'Distribution of AP Scores (n={len(image_ap_scores)} with defects)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
map_plot_path = 'checkpoints/test_map_analysis.png'
plt.savefig(map_plot_path, dpi=300, bbox_inches='tight')
print(f" Графики сохранены: {map_plot_path}")
plt.show()

os.makedirs('stage1-DeepLabv3-rn50-predictions', exist_ok=True)

#  СОЗДАНИЕ МАППИНГА ИНДЕКСОВ

# Создаём словарь: idx -> (метрика_значение, метрика_тип)
idx_to_metrics = {}

for metric_idx, batch_idx in enumerate(indices_with_defects):
    idx_to_metrics[batch_idx] = {
        'iou': iou_scores_with_defects[metric_idx],
        'dice': dice_scores_with_defects[metric_idx],
        'type': 'defect'
    }

for metric_idx, batch_idx in enumerate(indices_without_defects):
    idx_to_metrics[batch_idx] = {
        'fp_rate': false_positive_scores[metric_idx],
        'type': 'no_defect'
    }

#  ВИЗУАЛИЗАЦИЯ

num_viz = min(544, len(test_dataset))

for idx in tqdm(range(num_viz), desc="Saving visualizations", total=num_viz):
    # Загружаем данные из test_dataset
    img, mask = test_dataset[idx]
    img = img.numpy()

    # Преобразуем в формат для отображения [C, H, W] -> [H, W, C]
    img_display = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

    # Получаем предсказание [1, 1, H, W] -> [H, W]
    pred = all_predictions[idx][0, 0]
    mask_display = (mask.numpy()[0] * 255).astype(np.uint8)

    # Создаём визуализацию
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Оригинальное изображение
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2. Ground Truth маска
    axes[1].imshow(mask_display, cmap='gray')
    axes[1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 3. Предсказанная маска
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # 4. Наложение (оригинал + предсказание красным цветом)
    overlay = img_display.copy()
    overlay[pred > 0.5] = [255, 0, 0]
    axes[3].imshow(overlay)

    # Используем правильные метрики для каждого типа изображения
    metrics = idx_to_metrics[idx]
    if metrics['type'] == 'defect':
        title_text = f"Overlay (IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f})"
        save_name = f'sample_{idx:04d}_iou_{metrics["iou"]:.4f}.png'
    else:
        title_text = f"Overlay (FP Rate: {metrics['fp_rate']:.4f})"
        save_name = f'sample_{idx:04d}_fp_{metrics["fp_rate"]:.4f}.png'

    axes[3].set_title(title_text, fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()

    save_path = f'stage1-DeepLabv3-rn50-predictions/{save_name}'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

import shutil
shutil.make_archive('stage1-DeepLabv3-rn50-predictions', 'zip', 'stage1-DeepLabv3-rn50-predictions')