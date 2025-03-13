import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ====================== CARREGAR IMATGES ======================

folder = './highway/input'
files = [f for f in os.listdir(folder) if f.startswith('in') and f.endswith('.jpg')]

images = []

for file in files:
    num = int(file[2:8])
    if 1051 <= num <= 1350:
        filename = os.path.join(folder, file)
        im_color = cv2.imread(filename)
        images.append(cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY))

print('Carga completa de imágenes.')

# ====================== SEPARACIÓ TRAIN/TEST ======================

im_train = images[:150]
im_test = images[150:300]

# ====================== CÀLCUL MITJANA I DESVIACIÓ TÍPICA ======================

M, N = im_train[0].shape
num_images = len(im_train)

mean_image = np.zeros((M, N), dtype=np.float64)
sd_image = np.zeros((M, N), dtype=np.float64)

for img in im_train:
    mean_image += img.astype(np.float64)
mean_image /= num_images

for img in im_train:
    sd_image += (img.astype(np.float64) - mean_image) ** 2
sd_image = np.sqrt(sd_image / num_images)

mean_image = mean_image.astype(np.uint8)
sd_image = sd_image.astype(np.uint8)

plt.figure(1)
plt.imshow(mean_image, cmap='gray')
plt.title('Imatge de la mitjana')

plt.figure(2)
plt.imshow(sd_image, cmap='gray')
plt.title('Imatge de la desviació estàndard')

# ====================== SEGMENTACIÓ BÀSICA ======================

thr = 40

segmentation_images = [(np.abs(img - mean_image) > thr).astype(np.uint8) * 255 for img in images]

plt.figure(3)
plt.imshow(segmentation_images[6], cmap='gray')
plt.title('Segmentació bàsica')

# ====================== SEGMENTACIÓ AVANÇADA ======================

a = 0.15 * sd_image
b = 5

filter_image = sd_image.copy()
filter_image[filter_image < 35] = 130

threshold = a + b
adjusted_mean = mean_image - filter_image

segmentation_images = []
for img in images:
    diff_image = (np.abs(img - adjusted_mean) > threshold).astype(np.uint8) * 255
    segmented = adjusted_mean - diff_image
    segmented[segmented > b] = 255
    segmentation_images.append(segmented)

plt.figure(4)
plt.subplot(1, 2, 1)
plt.imshow(segmentation_images[6], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(images[6], cmap='gray')
plt.suptitle('Segmentació avançada: primera aproximació')

plt.show()
