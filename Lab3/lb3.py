import cv2
import numpy as np
from matplotlib import pyplot as plt

def remove_noise_and_sharpen(image_path, noise_reduction_method, sharpen_method):
   
    image = 255 - cv2.imread(image_path)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    denoised = apply_noise_reduction(image, method=noise_reduction_method)
    denoised_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    
    sharpened = apply_sharpening(denoised, method=sharpen_method)
    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    visualize_results(image_rgb, denoised_rgb, sharpened_rgb, 
                     noise_reduction_method, sharpen_method)
    
    return sharpened

def apply_noise_reduction(image, method):

    denoised = cv2.GaussianBlur(image, (7, 7), 0) #cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    return denoised

def apply_sharpening(image, method):
    
    kernel = np.array([[-1, -1, -1],
                        [-1, 12, -1],
                        [-1, -1, -1]]) / 4.0
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened

def visualize_results(original, denoised, sharpened, noise_method, sharpen_method):

    plt.figure(figsize=(20, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(original)
    plt.title('Исходное изображение')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(denoised)
    plt.title(f'После шумоподавления\n({noise_method} filter)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(sharpened)
    plt.title(f'После повышения четкости\n({sharpen_method} filter)')
    plt.axis('off')
    
    plt.show()
    

if __name__ == "__main__":
    image_path = './lb3png.jpg' 
    result = remove_noise_and_sharpen(image_path, 
                                    noise_reduction_method='nlmeans', 
                                    sharpen_method='highboost')

