import cv2
import matplotlib.pyplot as plt

def advanced_line_removal(image_path, output_path):
    # Загрузка и предобработка изображения
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Бинаризация для выделения линий и текста
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Удаление горизонтальных линий
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    detected_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, 
                                           horizontal_kernel, iterations=1)
    
    # Удаление вертикальных линий  
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, 
                                         vertical_kernel, iterations=2)
    
    # Комбинированная маска линий
    lines_mask = detected_horizontal + detected_vertical
    
    # Восстановление изображения
    result = gray.copy()
    result[lines_mask == 255] = 255  # Закрашивание линий белым
    
    # Дополнительная обработка для удаления шума
    result = cv2.medianBlur(result, 3)
    
    # Сохранение результата
    cv2.imwrite(output_path, result)

    # Создаем subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Показываем исходное изображение
    ax1.imshow(img)
    ax1.set_title('Исходное изображение')
    ax1.axis('off')

    # Показываем обработанное изображение
    img2 = cv2.imread(output_path)
    ax2.imshow(img2)
    ax2.set_title('Обработанное изображение')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()
    
    return result

# Использование
input_image = "textlb4.jpg"
output_image = "cleaned_text.jpg"
result = advanced_line_removal(input_image, output_image)