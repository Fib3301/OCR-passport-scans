import cv2
import numpy as np
import json
import os
from pathlib import Path
from utils import resize_image

def draw_boxes(image_path, json_path, output_path):
    """
    Отрисовывает боксы на изображении на основе данных из JSON файла
    
    Args:
        image_path (str): Путь к изображению
        json_path (str): Путь к JSON файлу с данными о боксах
        output_path (str): Путь для сохранения результата
    """
    # Загрузка изображения
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Не удалось прочитать изображение: {image_path}")
        return
    
    # Изменение размера изображения
    image = resize_image(image, target_width=800)
    
    # Загрузка данных из JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Создаем копию изображения для рисования
    result_image = image.copy()
    
    # Цвета для разных типов полей
    colors = {
        "surname": (0, 255, 0),      # Зеленый
        "name": (0, 255, 255),       # Желтый
        "patronymic": (255, 0, 0),   # Синий
        "birth_date": (255, 0, 255), # Пурпурный
        "gender": (0, 0, 255),       # Красный
        "birth_place": (255, 255, 0), # Голубой
        "passport_number": (128, 128, 128), # Серый
        "photo": (255, 128, 0)       # Оранжевый
    }
    
    # Отрисовка боксов
    for field in data['fields']:
        field_type = field['type']
        bbox = field['bbox']
        x, y, w, h = bbox
        
        # Выбираем цвет для типа поля
        color = colors.get(field_type, (255, 255, 255))  # Белый по умолчанию
        
        # Рисуем прямоугольник
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        
        # Добавляем текст с типом поля
        cv2.putText(result_image, field_type, (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Сохранение результата
    cv2.imwrite(str(output_path), result_image)
    print(f"Результат сохранен в {output_path}")

def process_directory(img_dir, correct_passport_data_dir, output_dir):
    """
    Обрабатывает все изображения в директории и создает визуализацию с идеальными прямоугольниками.
    
    Args:
        img_dir (str): Директория с изображениями
        correct_passport_data_dir (str): Директория с JSON файлами
        output_dir (str): Директория для сохранения результатов
    """
    img_path = Path(img_dir)
    correct_passport_data_path = Path(correct_passport_data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Получаем список всех изображений
    image_extensions = ('.jpg', '.jpeg', '.png')
    images = [img for img in img_path.glob('*') if img.suffix.lower() in image_extensions]
    
    if not images:
        print(f"Изображения не найдены в директории {img_dir}")
        return
    
    # Обработка каждого изображения
    for image_path in images:
        try:
            print(f"Обработка файла: {image_path.name}")
            
            # Формируем путь к соответствующему JSON файлу
            json_path = correct_passport_data_path / f"{image_path.stem}_passport.json"
            
            if not json_path.exists():
                print(f"Предупреждение: JSON файл не найден для {image_path.name}")
                continue
            
            # Формируем путь для сохранения результата
            output_file = output_path / f"{image_path.stem}_ground_truth.jpg"
            
            # Отрисовываем боксы
            draw_boxes(image_path, json_path, output_file)
            
        except Exception as e:
            print(f"Ошибка при обработке {image_path.name}: {str(e)}")
            continue

if __name__ == "__main__":
    # Пути к директориям
    img_dir = "img"
    correct_passport_data_dir = "correct_passport_data"
    output_dir = "ground_truth_visualization"
    
    # Запускаем обработку
    process_directory(img_dir, correct_passport_data_dir, output_dir) 