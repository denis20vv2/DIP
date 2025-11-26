import sys
sys.path.append('../')
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import matplotlib.pyplot as plt
from utility import segmentation_utils


image = cv.imread('./girl.jpg')
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

blur_image = cv.medianBlur(image_hsv, 3)
flat_image = np.float32(blur_image.reshape((-1,3)))
bandwidth = estimate_bandwidth(flat_image, quantile=.05, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

segments, inverse = np.unique(labeled, return_inverse=True)
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros((segments.shape[0], 3), dtype=float)

for i, idx in enumerate(inverse):
    total[idx] += flat_image[i]
    count[idx] += 1

avg = np.uint8(total / count)
mean_shift_image = avg[inverse].reshape(image_hsv.shape)


seeds = [(600, 550), (870, 500), (800, 450), (1600, 585), (1550, 535),
         (700, 750), (600, 350), (1000, 630), (630, 780), (600, 475), (700, 650)]
threshold = 15

segmented_region = segmentation_utils.region_growingHSV(mean_shift_image, seeds, threshold)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
segmented_region_dilated = cv.dilate(segmented_region, kernel, iterations=1)
segmented_region_smooth = cv.GaussianBlur(segmented_region_dilated, (5,5), 0)
_, segmented_region_final = cv.threshold(segmented_region_smooth, 50, 255, cv.THRESH_BINARY)

result = cv.bitwise_and(image, image, mask=segmented_region_final)

x = list(map(lambda x: x[1], seeds))
y = list(map(lambda x: x[0], seeds))

plt.figure(figsize=(15,20))
plt.subplot(1,2,1)
plt.scatter(x, y, marker="x", color="red", s=200)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение с seed-точками")
plt.subplot(1,2,2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title(f"Region Growing сглажено (threshold={threshold})")
plt.show()

plt.figure(figsize=(10,8))
plt.imshow(segmented_region_smooth, cmap='gray')
plt.title("Сглаженная маска Region Growing")
plt.colorbar()
plt.show()

binary_image = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
distance_map = ndimage.distance_transform_edt(binary_image)
local_max_coords = peak_local_max(distance_map, min_distance=20, labels=binary_image)
local_max_mask = np.zeros(distance_map.shape, dtype=bool)
local_max_mask[tuple(local_max_coords.T)] = True
markers = ndimage.label(local_max_mask, structure=np.ones((3,3)))[0]
labels_ws = watershed(-distance_map, markers, mask=binary_image)

mask1 = np.zeros(image.shape[0:2], dtype="uint8")
for label in np.unique(labels_ws):
    if label < 2:
        continue
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels_ws==label] = 255
    mask1 = cv.bitwise_or(mask1, mask)

mask1 = cv.dilate(mask1, cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3)), iterations=1)
mask1 = cv.GaussianBlur(mask1, (5,5), 0)

result_ws = cv.bitwise_and(image, image, mask=mask1)

plt.figure(figsize=(15,20))
plt.subplot(1,2,1)
plt.imshow(mask1, cmap='gray')
plt.title("Сглаженная маска Watershed")
plt.subplot(1,2,2)
plt.imshow(cv.cvtColor(result_ws, cv.COLOR_BGR2RGB))
plt.title("Результат Watershed сглажено")
plt.show()

pixels = gray.reshape(-1,1)
K = 3
kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
labels_k = kmeans.fit_predict(pixels)
cluster_centers = kmeans.cluster_centers_
segments_k = np.uint8(cluster_centers[labels_k].reshape(gray.shape))
segments_k[segments_k==cluster_centers.max()] = 0

segments_k = cv.dilate(segments_k, kernel, iterations=1)
segments_k = cv.GaussianBlur(segments_k, (15,15), 0)
result_k = cv.bitwise_and(gray, gray, mask=segments_k)

plt.figure(figsize=(15,20))
plt.subplot(1,3,1)
plt.imshow(cv.cvtColor(gray, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.subplot(1,3,2)
plt.imshow(segments_k, cmap='Set3')
plt.title("K-means сегментация сглажено")
plt.subplot(1,3,3)
plt.imshow(cv.cvtColor(result_k, cv.COLOR_BGR2RGB))
plt.title("Результат K-means сглажено")
plt.show()
