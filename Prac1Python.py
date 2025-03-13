import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import rectangle, erosion, dilation
from skimage.io import imsave
import imageio

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

def compute_mean_and_sd(im_list):
    # Convertim la llista d'imatges en un array 3D (pila d'imatges)
    images_stack = np.stack(im_list, axis=-1)
    
    # Càlcul de la mitjana i la desviació estàndard
    mean_image = np.mean(images_stack, axis=-1).astype(np.uint8)
    sd_image = np.std(images_stack, axis=-1).astype(np.uint8)
    
    # Mostrem les imatges
    plt.figure(1)
    plt.imshow(mean_image, cmap='gray')
    plt.title('Imatge de la mitjana')
    plt.axis('off')
    
    plt.figure(2)
    plt.imshow(sd_image, cmap='gray')
    plt.title('Imatge de la desviació estàndard')
    plt.axis('off')
    
    plt.show()
    
    return mean_image, sd_image

# ====================== SEGMENTACIÓ BÀSICA ======================

def segment_basic(im_list, mean_image):
    thr = 40
    
    segmentation_images = []
    for img in im_list:
        segmentation_images.append(np.abs(img - mean_image) > thr)
    
    # Imatge de prova
    plt.figure(3)
    plt.imshow(segmentation_images[6], cmap='gray')  # Índex 6 perquè Python comença en 0
    plt.title('Segmentació bàsica')
    plt.axis('off')
    plt.show()
    
    return segmentation_images


# ====================== SEGMENTACIÓ AVANÇADA ======================

def segment_images(im_list, mean_image, sd_image):
    a = 0.15 * sd_image
    b = 5
    
    filter_image = np.copy(sd_image)
    filter_image[filter_image < 35] = 130
    
    threshold = a + b
    adjusted_mean = mean_image - filter_image
    
    segmentation_images = []
    for img in im_list:
        diff_image = np.abs(img - adjusted_mean) > threshold
        segmented_img = adjusted_mean - (diff_image.astype(np.uint8) * 255)
        segmented_img[segmented_img > b] = 255
        segmentation_images.append(segmented_img)
    
    # Mostrem la imatge segmentada i la imatge original
    plt.figure(4)
    plt.subplot(1, 2, 1)
    plt.imshow(segmentation_images[6], cmap='gray')
    plt.title('Segmentació')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(im_list[6], cmap='gray')
    plt.title('Imatge original')
    plt.axis('off')
    
    plt.suptitle('Segmentació avançada: primera aproximació')
    plt.show()
    
    return segmentation_images

# ===================== OPENING ============================

def apply_opening(im_list, segmentation_images):
    SE = rectangle(3, 3)  # Element estructurant rectangular 3x3
    
    opened_images = []
    for seg_img in segmentation_images:
        opened_img = dilation(erosion(seg_img, SE), SE)
        opened_images.append(opened_img)
    
    # Mostrem una imatge abans i després de l'operació
    plt.figure(5)
    plt.subplot(1, 2, 1)
    plt.imshow(segmentation_images[6], cmap='gray')
    plt.title('Abans')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(opened_images[6], cmap='gray')
    plt.title('Després')
    plt.axis('off')
    
    plt.suptitle('Opening: Before and after')
    plt.show()
    
    return opened_images

# ===================== VIDEO AMB TEST IMAGES ============================

def save_video(images, filename='video_output.avi', fps=20):
    with imageio.get_writer(filename, fps=fps) as writer:
        for img in images:
            writer.append_data((img * 255).astype(np.uint8))
            
            
mean_image, sd_image =  compute_mean_and_sd(im_test)
segmentation_images = segment_images(im_test, mean_image, sd_image)
opened_images = apply_opening(im_test, segmentation_images)
save_video(opened_images)


# ========================= AVALUACIÓ ====================================

def evaluate_segmentation(opened_images_test, images_gt):
    accuracy = []
    
    for segmented, gt in zip(opened_images_test, images_gt):
        gt = (gt > 0).astype(np.uint8) * 255
        
        TP = np.sum((segmented == 255) & (gt == 255))
        TN = np.sum((segmented == 0) & (gt == 0))
        FP = np.sum((segmented == 255) & (gt == 0))
        FN = np.sum((segmented == 0) & (gt == 255))
        
        total_pixels = segmented.size
        correct_predictions = TP + TN
        
        accuracy.append(correct_predictions / total_pixels)
    
    mean_accuracy = np.mean(accuracy)
    print(f'Accuracy mitjà: {mean_accuracy:.4f}')
    return mean_accuracy

