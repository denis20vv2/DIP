import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

image_path = r"D:\PSUFINAL\COI\lenna.png"

if not os.path.exists(image_path):
    print("Файл не найден:", image_path)
    exit()

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


if image is None:
    print("Не удалось загрузить изображение:", image_path)
    exit()


hist = cv2.calcHist([image], [0], None, [256], [0, 256])
lut = np.array([255 * np.sum(hist[:i+1]) / np.sum(hist) for i in range(256)], dtype=np.uint8)
equalized_image = lut[image]


gs = plt.GridSpec(2, 2)
plt.figure(figsize=(12, 8))

plt.subplot(gs[0, 0])
plt.imshow(image, cmap='gray')
plt.title('Исходное изображение')
plt.axis('off')

plt.subplot(gs[0, 1])  
plt.imshow(equalized_image, cmap='gray')
plt.title('После эквализации')
plt.axis('off')

plt.subplot(gs[1, 0])
plt.hist(image.reshape(-1), 256, [0, 256], color='blue', alpha=0.7)
plt.title('Гистограмма исходного')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')

plt.subplot(gs[1, 1])
plt.hist(equalized_image.reshape(-1), 256, [0, 256], color='red', alpha=0.7)
plt.title('Гистограмма после эквализации')
plt.xlabel('Яркость')
plt.ylabel('Количество пикселей')

plt.tight_layout()
plt.show()

print(f"Исходное изображение: мин={np.min(image)}, макс={np.max(image)}")
print(f"После эквализации: мин={np.min(equalized_image)}, макс={np.max(equalized_image)}")