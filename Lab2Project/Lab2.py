import numpy as np
import cv2
from matplotlib import pyplot as plt

def histogram_equalization(image):

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    cdf = hist.cumsum()
    
    cdf_normalized = np.zeros_like(cdf, dtype=np.float64)
    
    mask = cdf > 0
    cdf_normalized[mask] = 255 * cdf[mask] / cdf[-1]
    
    lut = np.round(cdf_normalized).astype(np.uint8)
    
    equalized = lut[image]
    
    return equalized, hist, cdf, lut

def compare_histograms(original, equalized):
    
    hist_orig, _ = np.histogram(original.flatten(), 256, [0, 256])
    hist_eq, _ = np.histogram(equalized.flatten(), 256, [0, 256])
    
    cdf_orig = hist_orig.cumsum()
    cdf_eq = hist_eq.cumsum()
    
    cdf_orig_normalized = cdf_orig / cdf_orig[-1]
    cdf_eq_normalized = cdf_eq / cdf_eq[-1]
    
    return hist_orig, hist_eq, cdf_orig_normalized, cdf_eq_normalized


if __name__ == "__main__":
    image = cv2.imread('./winter_cat.png')
    
    equalized, hist_orig, cdf_orig, lut = histogram_equalization(image)

    hist_orig, hist_eq, cdf_orig_norm, cdf_eq_norm = compare_histograms(image, equalized)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.bar(range(256), hist_orig, color='blue', alpha=0.7)
    plt.title('Гистограмма исходного изображения')
    plt.xlabel('Уровень яркости')
    plt.ylabel('Частота')
    
    plt.subplot(2, 3, 3)
    plt.plot(cdf_orig_norm, color='red')
    plt.title('Кумулятивная гистограмма')
    plt.xlabel('Уровень яркости')
    plt.ylabel('Нормализованная частота')
    
    plt.subplot(2, 3, 4)
    plt.imshow(equalized, cmap='gray')
    plt.title('Эквализированное изображение')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.bar(range(256), hist_eq, color='green', alpha=0.7)
    plt.title('Гистограмма после эквализации')
    plt.xlabel('Уровень яркости')
    plt.ylabel('Частота')
    
    plt.subplot(2, 3, 6)
    plt.plot(cdf_eq_norm, color='red')
    plt.title('Кумулятивная гистограмма после эквализации')
    plt.xlabel('Уровень яркости')
    plt.ylabel('Нормализованная частота')
    
    plt.tight_layout()
    plt.show()
    
    print("Статистика обработки:")
    print(f"Минимальная яркость исходного: {image.min()}")
    print(f"Максимальная яркость исходного: {image.max()}")
    print(f"Минимальная яркость после эквализации: {equalized.min()}")
    print(f"Максимальная яркость после эквализации: {equalized.max()}")
    print(f"Уникальные уровни яркости исходного: {len(np.unique(image))}")
    print(f"Уникальные уровни яркости после эквализации: {len(np.unique(equalized))}")
