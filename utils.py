"""
Вспомогательные функции для обработки изображений
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any, Optional

def determine_field_type(index: int, total_fields: int) -> str:
    """
    Определяет тип поля на основе его позиции
    
    Args:
        index: Индекс текущего поля
        total_fields: Общее количество полей
        
    Returns:
        str: Тип поля
    """
    if total_fields >= 6:
        field_types = {
            0: "surname",
            1: "name",
            2: "patronymic",
            3: "birth_date",
            4: "gender"
        }
        return field_types.get(index, "birth_place")
    
    return f"field_{index+1}"

def save_results(output_path: Path, image_name: str, result_image: np.ndarray, 
                passport_data: Dict[str, Any]) -> None:
    """
    Сохраняет результаты обработки
    
    Args:
        output_path: Путь для сохранения результатов
        image_name: Имя исходного изображения
        result_image: Изображение с отрисованными контурами
        passport_data: Данные паспорта в формате JSON
    """
    output_path.mkdir(exist_ok=True)
    
    # Сохранение изображения с контурами
    output_file = output_path / f"{image_name}_contours.jpg"
    cv2.imwrite(str(output_file), result_image)
    
    # Сохранение JSON файла
    json_file = output_path / f"{image_name}_passport.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(passport_data, f, ensure_ascii=False, indent=2)

def load_ground_truth(ground_truth_path: Path) -> Optional[Dict[str, Any]]:
    """
    Загружает данные ground truth из JSON файла
    
    Args:
        ground_truth_path: Путь к файлу с ground truth данными
        
    Returns:
        Optional[Dict[str, Any]]: Данные ground truth или None, если файл не найден
    """
    if not ground_truth_path.exists():
        return None
        
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def resize_image(image: np.ndarray, target_width: int) -> np.ndarray:
    """
    Приведение изображения к единому разрешению с сохранением пропорций
    
    Args:
        image: Входное изображение
        target_width: Желаемая ширина
        
    Returns:
        np.ndarray: Измененное изображение
    """
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    return cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_AREA)

def process_image(image_path: Path, config: Dict[str, Any], output_path: Optional[Path] = None) -> np.ndarray:
    """
    Обработка изображения: чтение, resize, CLAHE, размытие, бинаризация
    
    Args:
        image_path: Путь к входному изображению
        config: Конфигурация параметров обработки
        output_path: Путь для сохранения результата
        
    Returns:
        np.ndarray: Бинарное изображение
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")
    image = resize_image(image, config['target_width'])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=config['clahe_clip_limit'],
        tileGridSize=config['clahe_grid_size']
    )
    gray_clahe = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, config['gaussian_kernel_size'], 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), binary)
    return binary 