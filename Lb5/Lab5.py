
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

image = cv.imread('./girl.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.show()

"""Отображаем разные каналы по разным осям на трехмерном графике. В случае модели RGB не видно кластеризации по цвету."""

r, g, b = cv.split(image_rgb)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = image_rgb.reshape((np.shape(image_rgb)[0]*np.shape(image_rgb)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")

image_hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)

h, s, v = cv.split(image_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
mask_green = cv.inRange(image_hsv, lower_green, upper_green)

lower_blouse = np.array([105, 40, 120])
upper_blouse = np.array([125, 120, 255])
mask_blouse = cv.inRange(image_hsv, lower_blouse, upper_blouse)


mask_total = cv.bitwise_or(mask_green, mask_blouse)

"""Применим обе маски и сгладим изображение"""

final_mask = mask_green + mask_blouse

final_result = cv.bitwise_and(image_rgb, image_rgb, mask=final_mask)
blur = cv.GaussianBlur(final_result, (7, 7), 0)

plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(final_mask, cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(final_result)
plt.subplot(1, 3, 3)
plt.imshow(blur)


