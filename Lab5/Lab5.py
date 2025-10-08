import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

# Загрузка изображения
image = plt.imread('oranges.jpg')

if image.dtype == np.uint8:
    image_normalized = image.astype(np.float32) / 255.0
else:
    image_normalized = image.astype(np.float32)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Исходное изображение')
axes[0].axis('off')

hsv_image = colors.rgb_to_hsv(image_normalized)

lower_orange = np.array([0.015, 0.35, 0.5])  

upper_orange = np.array([0.5, 1.0, 1.0])   

mask = np.all((hsv_image >= lower_orange) & (hsv_image <= upper_orange), axis=-1)


axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Маска оранжевых областей')
axes[1].axis('off')

highlighted_image = image.copy()
highlighted_image[~mask] = highlighted_image[~mask] * 0.3  # Затемнение не-оранжевых областей

axes[2].imshow(highlighted_image)
axes[2].set_title('Выделенные апельсины')
axes[2].axis('off')

plt.tight_layout()
plt.show()