import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

image = io.imread('oranges.jpg')

threshold = 0.4 

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(image)
axes[0, 0].set_title('Исходное изображение')

pixels = image.reshape(-1, 3)
scaler = StandardScaler()
pixels_normalized = scaler.fit_transform(pixels)

n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(pixels_normalized)

segmented_image = kmeans.cluster_centers_[labels]
segmented_image = scaler.inverse_transform(segmented_image)
segmented_image = segmented_image.reshape(image.shape).astype(np.uint8)

axes[0, 1].imshow(segmented_image)
axes[0, 1].set_title(f'K-means сегментация ({n_clusters} кластеров)')

cluster_colors = []
for i in range(n_clusters):
    cluster_pixels = pixels[labels == i]
    mean_color = cluster_pixels.mean(axis=0)
    cluster_colors.append(mean_color)

cluster_colors = np.array(cluster_colors)

print("Средние цвета всех кластеров:")
for i, color in enumerate(cluster_colors):
    print(f"Кластер {i}: RGB({color[0]:.0f}, {color[1]:.0f}, {color[2]:.0f})")

def get_orange_yellow_score(r, g, b):
    
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    base_score = r_norm * (1 - b_norm)

    if g > 0:
        rg_ratio = r / g
        if 0.8 <= rg_ratio <= 2.2:
            ratio_score = 1.0 - abs(rg_ratio - 1.2) / 1.4
        else:
            ratio_score = 0.0
    else:
        ratio_score = 0.0
    
    max_channel = max(r_norm, g_norm, b_norm)
    min_channel = min(r_norm, g_norm, b_norm)
    saturation_score = max_channel - min_channel
    
    final_score = base_score * 0.4 + ratio_score * 0.3 + saturation_score * 0.1
    return final_score

scores = []
for color_vec in cluster_colors:
    r, g, b = color_vec
    score = get_orange_yellow_score(r, g, b)
    scores.append(score)

print("\nОценки пригодности кластеров:")
for i, score in enumerate(scores):
    print(f"Кластер {i}: {score:.3f}")

selected_clusters = [i for i, score in enumerate(scores) if score > threshold]

print(f"\nПорог: {threshold}")
print(f"Выбранные кластеры: {selected_clusters}")
print(f"Количество выбранных кластеров: {len(selected_clusters)}")

combined_mask = np.zeros(image.shape[:2], dtype=bool)
for cluster_id in selected_clusters:
    cluster_mask = (labels == cluster_id).reshape(image.shape[:2])
    combined_mask = combined_mask | cluster_mask

axes[1, 0].imshow(combined_mask, cmap='gray')
axes[1, 0].set_title(f'Общая бинарная маска\n{len(selected_clusters)} кластеров')

highlighted = image.copy()
highlighted[~combined_mask] = highlighted[~combined_mask] * 0.3

axes[1, 1].imshow(highlighted)
axes[1, 1].set_title('Выделенные апельсины')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()

mask_image = (combined_mask * 255).astype(np.uint8)
io.imsave('mask.png', mask_image, check_contrast=False)
io.imsave('segments.jpg', segmented_image, check_contrast=False)
io.imsave('result.jpg', highlighted, check_contrast=False)
