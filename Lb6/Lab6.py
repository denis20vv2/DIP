# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:38:53 2023

@author: AM4
"""

import sys
sys.path.append('../')
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import data
from scipy import ndimage
import matplotlib.pyplot as plt
from utility import segmentation_utils


image = cv.imread('./girl.jpg')
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

seeds = [(600, 550), (870, 500), (800, 450), (1600, 585), (1550, 535), (700, 750), (600, 350), (1000, 630), (630, 780) , (600, 475), (700, 650)]

x = list(map(lambda x: x[1], seeds))
y = list(map(lambda x: x[0], seeds))
threshold = 71  

print("Точки для region growing:")
for i, (y_coord, x_coord) in enumerate(seeds):
    print(f"Точка {i+1}: ({x_coord}, {y_coord}) - HSV: {image_hsv[y_coord, x_coord]}")

segmented_region = segmentation_utils.region_growingHSV(image_hsv, seeds, threshold)

print(f"Уникальные значения в маске: {np.unique(segmented_region)}")
print(f"Размер маски: {segmented_region.shape}")

# накладываем маску - отображаем только участки попавшие в какой-либо сегмент
result = cv.bitwise_and(image, image, mask=segmented_region)

# отображаем полученное изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.scatter(x, y, marker="x", color="red", s=200)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение с точками роста")
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title(f"Region Growing (threshold={threshold})")
plt.show()

# Визуализируем маску отдельно
plt.figure(figsize=(10, 8))
plt.imshow(segmented_region, cmap='gray')
plt.title("Маска Region Growing")
plt.colorbar()
plt.show()

# Разделение областей
qt = segmentation_utils.QTree(stdThreshold = 0.25, minPixelSize = 4, img = image.copy()) 
qt.subdivide()
tree_image = qt.render_img(thickness=1, color=(0,0,0))

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(tree_image, cv.COLOR_BGR2RGB))
plt.title("Quadtree сегментация")
plt.show()


# Алгоритм водораздела
# Бинаризируем изображение
binary_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
# Определяем карту расстояний
distance_map = ndimage.distance_transform_edt(binary_image)
# Определяем локальные максимумы
local_max_coords = peak_local_max(distance_map, min_distance=20, labels=binary_image)
local_max_mask = np.zeros(distance_map.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
# 4 Каждому минимуму присваивается метка и начинается заполнение бассейнов метками
markers = ndimage.label(local_max_mask, structure=np.ones((3, 3)))[0]
labels = watershed(-distance_map, markers, mask=binary_image)
# построим результаты работы алгоритма
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap="gray")
plt.title("Бинарное изображение")
plt.subplot(1, 3, 2)
plt.imshow(np.uint8(distance_map + 50), cmap="gray")
plt.title("Карта расстояний")
plt.subplot(1, 3, 3)
plt.imshow(np.uint8(labels))
plt.title("Watershed сегментация")
plt.show()

# Найдем границы контуров и положим в маску все кроме метки 0
mask1 = np.zeros(image.shape[0:2], dtype="uint8")
total_area = 0
image_with_contours = image.copy()  # создаем копию для рисования контуров

for label in np.unique(labels):
    if label < 2:
        continue
    # Create a mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    mask1 = mask1 + mask

    # Find contours and determine contour area
    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if cnts:  # проверяем, что контуры найдены
        c = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c)
        total_area += area
        cv.drawContours(image_with_contours, [c], -1, (36,255,12), 1)

result = cv.bitwise_and(image, image, mask=mask1)

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(mask1, cmap="gray")
plt.title("Маска Watershed")
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title("Результат Watershed")
plt.show()

## Методы кластеризации. K-средних
# Преобразуем изображение в оттенках серого в одномерный массив
pixels = gray.reshape(-1, 1)
# Задаем число кластеров для сегментации
K = 3
# С помощью библиотеки sklearn.cluster import KMeans проводим кластеризацию по яркости
kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
labels = kmeans.fit_predict(pixels)
cluster_centers = kmeans.cluster_centers_
print("Центры кластеров:", np.uint8(cluster_centers))
# Каждому пикселю назначаем значение из центра кластера
segments = np.uint8(cluster_centers[labels].reshape(gray.shape))
# Удалим самые яркие пиксели
segments[segments==167] = 0
result = cv.bitwise_and(gray, gray, mask=segments)
# Отобразим избражения 
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.subplot(1, 3, 2)
plt.imshow(segments, cmap='Set3')
plt.title("K-means сегментация")
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title("Результат K-means")
plt.show()


## Методы кластеризации. Сдвиг среднего (Mean shift)
# Сглаживаем чтобы уменьшить шум
blur_image = cv.medianBlur(image, 3)
# Выстраиваем пиксели в один ряд и переводим в формат с плавающей точкой
flat_image = np.float32(blur_image.reshape((-1,3)))

# Используем meanshift из библиотеки sklearn
bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

# получим количество сегментов
segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])

# получим средний цвет сегмента
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)

# Для каждого пикселя проставим средний цвет его сегмента
mean_shift_image = avg[labeled].reshape((image.shape))
# Маской скроем один из сегментов
mask1 = mean_shift_image[:,:,0]
mask1[mask1==89] = 0
mean_shift_with_mask_image = cv.bitwise_and(image, image, mask=mask1)
# Построим изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.subplot(1, 3, 2)
plt.imshow(mean_shift_image, cmap='Set3')
plt.title("Mean Shift сегментация")
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(mean_shift_with_mask_image, cv.COLOR_BGR2RGB))
plt.title("Результат Mean Shift")
plt.show()