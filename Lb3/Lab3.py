import cv2
import numpy as np
import matplotlib.pyplot as plt


def getPSNR(I1, I2):
    mse = np.mean((I1.astype(np.float32) - I2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    return 10 * np.log10((255.0 ** 2) / mse)

def getSSIM(img1, img2):
    C1 = 6.5025
    C2 = 58.5225
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    sigma1_2 = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1 * mu1
    sigma2_2 = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2 * mu2
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_2 + sigma2_2 + C2))
    return cv2.mean(ssim_map)[0]


def add_gauss_noise(img, mean=0, sigma=0.15):
    gauss = np.random.normal(mean, sigma*255, img.shape).astype(np.float32)
    noisy = cv2.add(img.astype(np.float32), gauss)
    return np.clip(noisy, 0, 255).astype(np.uint8)


final_rotated_image = cv2.imread('./Image-68-1c19cc.jpg')

neg_image = 255 - final_rotated_image

kernel55 = np.ones((5, 5), np.float32) / 25
kernel77 = np.ones((7, 7), np.float32) / 49

channels = cv2.split(neg_image)
noisy_channels = [add_gauss_noise(ch, 0, 0.15) for ch in channels]
filtered_channels1 = [cv2.filter2D(ch, -1, kernel55) for ch in noisy_channels]
filtered_channels2 = [cv2.filter2D(ch, -1, kernel77) for ch in noisy_channels]
gaussian_channels1 = [cv2.GaussianBlur(ch, (7,7), 0) for ch in noisy_channels]
gaussian_channels2 = [cv2.GaussianBlur(ch, (15,15), 0) for ch in noisy_channels]

noisy_image = cv2.merge(noisy_channels)
filtered_image1 = cv2.merge(filtered_channels1)
filtered_image2 = cv2.merge(filtered_channels2)
gaussian_image1 = cv2.merge(gaussian_channels1)
gaussian_image2 = cv2.merge(gaussian_channels2)

gs = plt.GridSpec(2, 3)
plt.figure(figsize=(15, 12))

plt.subplot(gs[0,0])
plt.xticks([]), plt.yticks([])
plt.title('Исходное изображение')
plt.imshow(cv2.cvtColor(final_rotated_image, cv2.COLOR_BGR2RGB))

plt.subplot(gs[1,0])
plt.xticks([]), plt.yticks([])
plt.title(f'Зашумленное\nPSNR={getPSNR(final_rotated_image,noisy_image):.2f}, SSIM={getSSIM(final_rotated_image,noisy_image):.2f}')
plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))

plt.subplot(gs[0,1])
plt.xticks([]), plt.yticks([])
plt.title(f'Фильтр 5x5\nPSNR={getPSNR(final_rotated_image,filtered_image1):.2f}, SSIM={getSSIM(final_rotated_image,filtered_image1):.2f}')
plt.imshow(cv2.cvtColor(filtered_image1, cv2.COLOR_BGR2RGB))

plt.subplot(gs[0,2])
plt.xticks([]), plt.yticks([])
plt.title(f'Фильтр 7x7\nPSNR={getPSNR(final_rotated_image,filtered_image2):.2f}, SSIM={getSSIM(final_rotated_image,filtered_image2):.2f}')
plt.imshow(cv2.cvtColor(filtered_image2, cv2.COLOR_BGR2RGB))

plt.subplot(gs[1,1])
plt.xticks([]), plt.yticks([])
plt.title(f'Гаусс 7x7\nPSNR={getPSNR(final_rotated_image,gaussian_image1):.2f}, SSIM={getSSIM(final_rotated_image,gaussian_image1):.2f}')
plt.imshow(cv2.cvtColor(gaussian_image1, cv2.COLOR_BGR2RGB))

plt.subplot(gs[1,2])
plt.xticks([]), plt.yticks([])
plt.title(f'Гаусс 15x15\nPSNR={getPSNR(final_rotated_image,gaussian_image2):.2f}, SSIM={getSSIM(final_rotated_image,gaussian_image2):.2f}')
plt.imshow(cv2.cvtColor(gaussian_image2, cv2.COLOR_BGR2RGB))

plt.show()

kernel1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
kernel2 = np.array([[-0.25,-0.25,-0.25],[-0.25,3,-0.25],[-0.25,-0.25,-0.25]])
kernel3 = np.array([[0,-0.25,0],[-0.25,2,-0.25],[0,-0.25,0]])

filtered_image1 = cv2.filter2D(neg_image, -1, kernel1)
filtered_image2 = cv2.filter2D(neg_image, -1, kernel2)
filtered_image3 = cv2.filter2D(neg_image, -1, kernel3)

psnr1, ssim1 = getPSNR(final_rotated_image, filtered_image1), getSSIM(final_rotated_image, filtered_image1)
psnr2, ssim2 = getPSNR(final_rotated_image, filtered_image2), getSSIM(final_rotated_image, filtered_image2)
psnr3, ssim3 = getPSNR(final_rotated_image, filtered_image3), getSSIM(final_rotated_image, filtered_image3)

plt.figure(figsize=(18, 8))
plt.subplot(141)
plt.title('Исходное изображение')
plt.imshow(cv2.cvtColor(neg_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(142)
plt.title(f'Фильтр 1\nPSNR={psnr1:.2f}, SSIM={ssim1:.2f}')
plt.imshow(cv2.cvtColor(filtered_image1, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(143)
plt.title(f'Фильтр 2\nPSNR={psnr2:.2f}, SSIM={ssim2:.2f}')
plt.imshow(cv2.cvtColor(filtered_image2, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(144)
plt.title(f'Фильтр 3\nPSNR={psnr3:.2f}, SSIM={ssim3:.2f}')
plt.imshow(cv2.cvtColor(filtered_image3, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()
