import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def read_image(path):
    img = Image.open(path)
    return img

def to_array(img):
    img_array = np.array(img, dtype=np.int16)
    return img_array

def to_image(img_array):
    return Image.fromarray(img_array.astype('uint8'))

def to_grayscale_simple(img_array):
    R = img_array[:, :, 0] #czerwony
    G = img_array[:, :, 1] #zielony
    B = img_array[:, :, 2] #niebieski
    gray_simple = (R + G + B) / 3
    return gray_simple

def to_grayscale(img_array):
    R = img_array[:, :, 0] #czerwony
    G = img_array[:, :, 1] #zielony
    B = img_array[:, :, 2] #niebieski
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray

def brightness(img_array, b):
    img_array_bright = img_array + b
    img_array_bright_clipped = np.clip(img_array_bright, 0, 255)
    return img_array_bright_clipped

def gamma_correction(img_array, gamma):
    img_array_gamma_corrected = 255 * (img_array / 255) ** gamma
    return img_array_gamma_corrected

def contrast_correction(img_array, c):
    img_array_contrast_corrected = 128 + c * (img_array - 128)
    img_array_contrast_corrected_clipped = np.clip(img_array_contrast_corrected, 0, 255)
    return img_array_contrast_corrected_clipped

def negative(img_array):
    img_array_negative = np.copy(img_array)
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array_negative[:, :, :3] = 255 - img_array[:, :, :3]
    else:
        img_array_negative = 255 - img_array
        
    return img_array_negative

def binarization(gray, threshold):
    gray_binary = (gray >= threshold) * 255
    return gray_binary

def padding(img_array, middle):
    if img_array.ndim == 3: 
        padded = np.pad(img_array, pad_width=((middle, middle), (middle, middle), (0, 0)), mode='edge')
    else:
        padded = np.pad(img_array, pad_width=((middle, middle), (middle, middle)), mode='edge')
    return padded

def apply_kernel(img_array, kernel):
    new_image_array = np.zeros_like(img_array)
    size = kernel.shape[0]
    middle = size // 2
    padded_image_array = padding(img_array, middle)
    if img_array.ndim == 3: 
        height, width, channels = img_array.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels): 
                    region = padded_image_array[i:i+size, j:j+size, k]
                    new_value = np.sum(region * kernel)
                    new_image_array[i, j, k] = new_value
                    
    else: 
        height, width = img_array.shape
        for i in range(height):
            for j in range(width):
                region = padded_image_array[i:i+size, j:j+size]
                new_value = np.sum(region * kernel)
                new_image_array[i, j] = new_value

    new_image_array = np.clip(new_image_array, 0, 255)            
    return new_image_array


def mean_kernel(size):
    kernel = np.ones((size, size)) / (size * size)
    return kernel

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    middle = size // 2
    for i in range(size):
        for j in range(size):
            x = i - middle
            y = j - middle
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def sharpening_kernel(size, amount):
    identity = np.zeros((size, size))
    identity[size // 2, size // 2] = 1
    blur = np.ones((size, size)) / (size * size)
    kernel = identity + amount * (identity - blur)
    return kernel

def save_image(img, path):
    img.save(path)
    return path

def histogram(img_array):
    if img_array.ndim == 2: # grayscale
        plt.hist(img_array.flatten(), bins=256, range=(0, 256), color='gray')
        plt.title("Histogram jasności")
        plt.xlabel("Wartość piksela (0-255)")
        plt.ylabel("Liczba pikseli")
        plt.yscale('log')
        #plt.show()
    if img_array.ndim == 3: # RGB
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            plt.hist(img_array[:, :, i].flatten(), bins=256, range=(0, 256), color=color, alpha=0.5)
        plt.title("Histogram kolorów")
        plt.xlabel("Wartość piksela (0-255)")
        plt.ylabel("Liczba pikseli")
        plt.yscale('log')
        plt.legend(colors)
        #plt.show()
    return plt

def line_rgb(img_array):
    hist_r, _ = np.histogram(img_array[:, :, 0].ravel(), bins=256, range=(0, 256))
    hist_g, _ = np.histogram(img_array[:, :, 1].ravel(), bins=256, range=(0, 256))
    hist_b, _ = np.histogram(img_array[:, :, 2].ravel(), bins=256, range=(0, 256))

    plt.figure(figsize=(10, 5))
    plt.title("Histogram RGB")
    plt.xlabel("Wartość piksela")
    plt.ylabel("Liczba pikseli")

    plt.plot(hist_r, color='red', label='Czerwony')
    plt.plot(hist_g, color='green', label='Zielony')
    plt.plot(hist_b, color='blue', label='Niebieski')

    plt.legend()
    plt.xlim([0, 255])
    plt.yscale('log')
    #plt.show()
    return plt

def vertical_projection(img_array):
    plt.figure(figsize=(10, 5))
    plt.title("Projekcja pionowa (Skala logarytmiczna)")
    plt.xlabel("Kolumna obrazu")
    plt.ylabel("Suma pikseli (log)")
    
    if img_array.ndim == 3:
        colors = ['red', 'green', 'blue']
        labels = ['Czerwony', 'Zielony', 'Niebieski']
        for i in range(3):
            projection = np.sum(img_array[:, :, i], axis=0)
            projection = np.where(projection == 0, 1, projection)
            plt.plot(projection, color=colors[i], label=labels[i], linewidth=1.5, alpha=0.7)
        plt.legend()
    else:
        projection = np.sum(img_array, axis=0)
        projection = np.where(projection == 0, 1, projection)
        plt.plot(projection, color='black', linewidth=1.5)
        plt.fill_between(range(len(projection)), projection, color='gray', alpha=0.3)
        
    plt.yscale('log')
    plt.xlim([0, img_array.shape[1]])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt

def horizontal_projection(img_array):
    plt.figure(figsize=(10, 5))
    plt.title("Projekcja pozioma (Skala logarytmiczna)")
    plt.xlabel("Wiersz obrazu")
    plt.ylabel("Suma pikseli (log)")
    
    if img_array.ndim == 3:
        colors = ['red', 'green', 'blue']
        labels = ['Czerwony', 'Zielony', 'Niebieski']
        for i in range(3):
            projection = np.sum(img_array[:, :, i], axis=1)
            projection = np.where(projection == 0, 1, projection)
            plt.plot(projection, color=colors[i], label=labels[i], linewidth=1.5, alpha=0.7)
        plt.legend()
    else:
        projection = np.sum(img_array, axis=1)
        projection = np.where(projection == 0, 1, projection)
        plt.plot(projection, color='black', linewidth=1.5)
        plt.fill_between(range(len(projection)), projection, color='gray', alpha=0.3)
        
    plt.yscale('log')
    plt.xlim([0, img_array.shape[0]])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    return plt

def Roberts_cross(gray):
    Gx = np.array([[1,0],[0,-1]])
    Gy = np.array([[0,1],[-1,0]])
    return edge_detection(gray, Gx, Gy)

def Sobel(gray):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [ 0, 0, 0], [ 1, 2, 1]])
    return edge_detection(gray, Kx, Ky)

def edge_detection(gray, Kx, Ky):
    size = Kx.shape[0]
    middle = size // 2
    padded = padding(gray, middle)
    sobel_final = np.zeros_like(gray)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            region = padded[i:i+size, j:j+size]
            Gx = np.sum(region * Kx)
            Gy = np.sum(region * Ky)
            sobel_final[i, j] = np.clip(np.sqrt(Gx**2 + Gy**2), 0, 255)
    return sobel_final

def check_kernel(kernel):
    if kernel.ndim != 2:
        raise ValueError("Kernel musi być macierzą 2D")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel musi mieć nieparzyste wymiary")
    return kernel

def any_filter(img_array, kernel):
    kernel = check_kernel(kernel)
    return apply_kernel(img_array, kernel)


def otsu_binarization(gray):
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total_pixels = gray.size
    
    current_max = 0
    best_threshold = 0
    sum_total = np.dot(np.arange(256), hist)
    
    sum_background = 0
    weight_background = 0
    
    for t in range(256):
        weight_background += hist[t]
        if weight_background == 0:
            continue
            
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
            
        sum_background += t * hist[t]
        
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        var_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if var_between > current_max:
            current_max = var_between
            best_threshold = t
            
    return (gray >= best_threshold) * 255

def kuwahara_filter(img_array, size):
    new_img = np.zeros_like(img_array)
    k = size // 2
    padded = padding(img_array, k)
    
    if img_array.ndim == 3:
        h, w, c = img_array.shape
        for i in range(h):
            for j in range(w):
                for ch in range(c):
                    r1 = padded[i:i+k+1, j:j+k+1, ch]
                    r2 = padded[i:i+k+1, j+k:j+2*k+1, ch]
                    r3 = padded[i+k:i+2*k+1, j:j+k+1, ch]
                    r4 = padded[i+k:i+2*k+1, j+k:j+2*k+1, ch]
                    
                    vars_list = [np.var(r1), np.var(r2), np.var(r3), np.var(r4)]
                    means_list = [np.mean(r1), np.mean(r2), np.mean(r3), np.mean(r4)]
                    
                    min_var_idx = np.argmin(vars_list)
                    new_img[i, j, ch] = means_list[min_var_idx]
    else:
        h, w = img_array.shape
        for i in range(h):
            for j in range(w):
                r1 = padded[i:i+k+1, j:j+k+1]
                r2 = padded[i:i+k+1, j+k:j+2*k+1]
                r3 = padded[i+k:i+2*k+1, j:j+k+1]
                r4 = padded[i+k:i+2*k+1, j+k:j+2*k+1]
                
                vars_list = [np.var(r1), np.var(r2), np.var(r3), np.var(r4)]
                means_list = [np.mean(r1), np.mean(r2), np.mean(r3), np.mean(r4)]
                
                min_var_idx = np.argmin(vars_list)
                new_img[i, j] = means_list[min_var_idx]
                
    return new_img

def wave_distortion(img_array, amplitude, wavelength):
    new_img = np.zeros_like(img_array)
    
    if img_array.ndim == 3:
        h, w, c = img_array.shape
        for i in range(h):
            offset_x = int(amplitude * np.sin(2 * np.pi * i / wavelength))
            for j in range(w):
                new_j = j + offset_x
                if 0 <= new_j < w:
                    new_img[i, j] = img_array[i, new_j]
    else:
        h, w = img_array.shape
        for i in range(h):
            offset_x = int(amplitude * np.sin(2 * np.pi * i / wavelength))
            for j in range(w):
                new_j = j + offset_x
                if 0 <= new_j < w:
                    new_img[i, j] = img_array[i, new_j]
                    
    return new_img


def median_filter(img_array, size):
    new_image_array = np.zeros_like(img_array)
    middle = size // 2
    padded_image_array = padding(img_array, middle)
    
    if img_array.ndim == 3:
        height, width, channels = img_array.shape
        for i in range(height):
            for j in range(width):
                for k in range(channels):
                    region = padded_image_array[i:i+size, j:j+size, k]
                    new_image_array[i, j, k] = np.median(region)

    else:
        height, width = img_array.shape
        for i in range(height):
            for j in range(width):
                region = padded_image_array[i:i+size, j:j+size]
                new_image_array[i, j] = np.median(region)
                
    return new_image_array

def Prewitt(gray):
    Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Ky = np.array([[-1, -1, -1], [ 0, 0, 0], [ 1, 1, 1]])
    return edge_detection(gray, Kx, Ky)

def Laplace(gray):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    laplacian = apply_kernel(gray, kernel)
    laplacian_clipped = np.clip(np.abs(laplacian), 0, 255)
    return laplacian_clipped


def add_salt_and_pepper(image_array, density=0.05):
    noisy_image = np.copy(image_array)
    rows, cols = image_array.shape[:2]
    prob_matrix = np.random.rand(rows, cols)
    white = [255, 255, 255] if image_array.ndim == 3 else 255
    noisy_image[prob_matrix < (density / 2)] = white
    black = [0, 0, 0] if image_array.ndim == 3 else 0
    noisy_image[prob_matrix > (1 - density / 2)] = black
    
    return noisy_image

