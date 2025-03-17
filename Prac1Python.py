import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import rectangle, erosion, dilation
from skimage.io import imsave
import imageio
from PIL import Image
from skimage.color import rgb2gray

# ====================== CARREGAR IMATGES ======================

folder = './highway/input'
files = [f for f in os.listdir(folder) if f.startswith('in') and f.endswith('.jpg')]

images = []

for file in files:
    num = int(file[2:8])
    if 1051 <= num <= 2050:
        filename = os.path.join(folder, file)
        im_color = Image.open(filename)

        # Convertir la imagen a escala de grises usando PIL
        im_gray = im_color.convert('L')
        
        # Convertir a un array de numpy
        im_gray = np.array(im_gray)
        
        # Agregar la imagen en escala de grises a la lista
        images.append(im_gray)

print('Carga completa de imágenes.')

# ====================== SEPARACIÓ TRAIN/TEST ======================

im_train = np.array(images[:150], dtype=np.float64)
im_test = np.array(images[150:], dtype=np.float64)

# ====================== CÀLCUL MITJANA I DESVIACIÓ TÍPICA ======================

def compute_mean_and_sd(im):
    # Obtener las dimensiones de la primera imagen
    M, N = im[0].shape
    
    # Número de imágenes
    num_images = len(im)
    
    # Convertir todo el array de imágenes a float64 de una vez
    im = np.array(im, dtype=np.float64)
    
    # Calcular la media usando el axis correcto (más eficiente)
    mean_image = np.mean(im, axis=0)
    
    # Calcular la desviación estándar usando la fórmula de MATLAB
    # MATLAB usa: sqrt(sum((X - mean(X)).^2) / (N-1))
    sd_image = np.sqrt(np.sum((im - mean_image) ** 2, axis=0) / (num_images - 1))
    
    # Mostrar las imágenes
    plt.figure(1)
    plt.imshow(mean_image, cmap='gray')
    plt.title('Imagen de la media')
    
    plt.figure(2)
    plt.imshow(sd_image, cmap='gray')
    plt.title('Imagen de la desviación estándar')
    
    plt.show()
    
    return mean_image, sd_image

# ====================== SEGMENTACIÓ BÀSICA ======================

def segment_basic(im_list, mean_image):
    thr = 20
    
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
    a = 1.5 * sd_image
    b = 5

    
    threshold = a + b
    
    segmentation_images = []
    for img in im_list:
        diff_image = np.abs(img) > threshold
        segmented_img = (diff_image.astype(np.uint8) * 255)
        segmented_img[segmented_img > b] = 255
        segmented_img = 255-segmented_img
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
    SE = rectangle(4, 4)  # Element estructurant rectangular 3x3
    
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
            # Asegurar que la imagen está en el rango correcto [0, 255]
            img = np.clip(img, 0, 1)  # Limitar los valores al rango [0,1]
            img = (img * 255).astype(np.uint8)  # Convertir a uint8
            
            # Si la imagen está en escala de grises, convertir a RGB
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)  # Convertir a formato (H, W, 3)
            
            writer.append_data(img)
            
mean_image, sd_image = compute_mean_and_sd(im_test)
segmentation_images = segment_images(im_test, mean_image, sd_image)
opened_images = apply_opening(im_test, segmentation_images)

# Guardar el video con las imágenes corregidas
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
