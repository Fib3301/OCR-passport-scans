"""
Модуль для обработки контуров и распознавания полей паспорта
"""

import cv2
import numpy as np
import pytesseract
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from config import MORPHOLOGY_CONFIG, CONTOUR_FILTER_CONFIG
from utils import determine_field_type

class ContourProcessor:
    """
    Класс для обработки контуров и распознавания полей паспорта
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация процессора контуров
        
        Args:
            config: Конфигурация параметров обработки
        """
        self.config = config or {}
    
    def apply_morphological_operations(self, binary_image: np.ndarray, 
                                     operation_type: str) -> List[np.ndarray]:
        """
        Применение морфологических операций к бинарному изображению
        
        Args:
            binary_image: Бинарное изображение
            operation_type: Тип операции ('owner_info', 'series_number')
            
        Returns:
            List[np.ndarray]: Список контуров
        """
        config = MORPHOLOGY_CONFIG[operation_type]
        
        # Создание ядер для морфологических операций
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                               config['close_kernel_size'])
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                              config['open_kernel_size'])
        
        closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_close)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        return contours

    def filter_contours(self, contours: List[np.ndarray], operation_type: str, 
                       image_width: int, image_height: int) -> List[np.ndarray]:
        """
        Фильтрация контуров на основе заданных критериев
        
        Args:
            contours: Список контуров
            operation_type: Тип операции
            image_width: Ширина изображения
            image_height: Высота изображения
            
        Returns:
            List[np.ndarray]: Отфильтрованные контуры
        """
        config = CONTOUR_FILTER_CONFIG[operation_type]
        filtered_contours = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if operation_type == 'owner_info':
                if (x >= image_width * config['x_min_ratio'] and 
                    x + w <= image_width * config['x_max_ratio']):
                    filtered_contours.append(contour)
                    
            elif operation_type == 'series_number':
                if (x >= image_width * config['x_min_ratio'] and 
                    h / w >= config['min_height_width_ratio']):
                    filtered_contours.append(contour)
        
        return filtered_contours

    def process_owner_information_contours(self, contours: List[np.ndarray], 
                                         image_width: int, image_height: int, 
                                         binary: np.ndarray, 
                                         padding: int = 5) -> Tuple[List[Dict], 
                                                                  List[Dict], 
                                                                  List[Dict]]:
        """
        Обработка контуров области информации о владельце паспорта
        
        Args:
            contours: Список контуров
            image_width: Ширина изображения
            image_height: Высота изображения
            binary: Бинарное изображение
            padding: Отступ для рисования прямоугольников
            
        Returns:
            Tuple[List[Dict], List[Dict], List[Dict]]: 
                (поля владельца, части места рождения, визуализация)
        """
        fields = []
        birth_place_parts = []
        visualization = []
        
        sorted_contours = sorted(contours, 
                               key=lambda contour: cv2.boundingRect(contour)[1])
        
        for i, contour in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = (x - padding, y - padding, w + 2*padding, h + 2*padding)
            
            roi = binary[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi, 
                                             config=self.config.get('ocr_config', '')).strip()
            
            field_type = determine_field_type(i, len(sorted_contours))
            
            if field_type == "birth_place":
                birth_place_parts.append({
                    "value": text,
                    "bbox": [x, y, w, h],
                    "y_position": y
                })
            else:
                fields.append({
                    "type": field_type,
                    "value": text,
                    "bbox": [x, y, w, h],
                })
            
            visualization.append({
                "bbox": (x, y, w, h),
                "type": field_type,
                "color": (0, 255, 0)
            })
        
        return fields, birth_place_parts, visualization

    def process_passport_number_contours(self, contours: List[np.ndarray], 
                                      binary: np.ndarray, 
                                      padding: int = 5) -> Tuple[List[Dict], 
                                                               List[Dict]]:
        """
        Обработка контуров области серии и номера паспорта
        
        Args:
            contours: Список контуров
            binary: Бинарное изображение
            padding: Отступ для рисования прямоугольников
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (поля номера паспорта, визуализация)
        """
        fields = []
        visualization = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = (x - padding, y - padding, w + 2*padding, h + 2*padding)
            
            roi = binary[y:y+h, x:x+w]
            roi_rotated = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            text = pytesseract.image_to_string(
                roi_rotated, 
                config='--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            ).strip()
            
            if text and text.isdigit():
                fields.append({
                    "type": "passport_number",
                    "value": text,
                    "bbox": [x, y, w, h],
                })
                
                visualization.append({
                    "bbox": (x, y, w, h),
                    "type": "passport_number",
                    "color": (0, 0, 255)
                })
        
        return fields, visualization

    def process_birth_place_data(self, birth_place_parts: List[Dict]) -> Optional[Dict]:
        """
        Обработка и объединение данных о месте рождения
        
        Args:
            birth_place_parts: Список частей места рождения
            
        Returns:
            Optional[Dict]: Информация о месте рождения
        """
        if not birth_place_parts:
            return None
            
        birth_place_parts.sort(key=lambda x: x["y_position"], reverse=True)
        birth_place_text = " ".join(reversed([part["value"] for part in birth_place_parts]))
        
        min_x = min(part["bbox"][0] for part in birth_place_parts)
        min_y = min(part["bbox"][1] for part in birth_place_parts)
        max_x = max(part["bbox"][0] + part["bbox"][2] for part in birth_place_parts)
        max_y = max(part["bbox"][1] + part["bbox"][3] for part in birth_place_parts)
        
        return {
            "type": "birth_place",
            "value": birth_place_text,
            "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
            "parts": birth_place_parts
        }

    def draw_visualization(self, image: np.ndarray, 
                         visualization_data: List[Dict]) -> np.ndarray:
        """
        Отрисовка визуализации на изображении
        
        Args:
            image: Исходное изображение
            visualization_data: Список данных для визуализации
            
        Returns:
            np.ndarray: Изображение с отрисованными контурами
        """
        result_image = image.copy()
        
        for item in visualization_data:
            x, y, w, h = item["bbox"]
            field_type = item["type"]
            color = item["color"]
            
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result_image, field_type, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result_image

    def process_contours(self, image: np.ndarray, binary: np.ndarray, 
                        padding: int = 5) -> Tuple[np.ndarray, 
                                                  Dict[str, Any]]:
        """
        Обработка контуров и их фильтрация
        
        Args:
            image: Исходное изображение
            binary: Бинарное изображение
            padding: Отступ для рисования прямоугольников
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 
                (изображение с отрисованными контурами, структура данных для JSON)
        """
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # Получаем контуры для разных областей
        owner_info_contours = self.apply_morphological_operations(binary, 'owner_info')
        filtered_owner_info = self.filter_contours(owner_info_contours, 'owner_info', 
                                                 image_width, image_height)
        
        series_number_contours = self.apply_morphological_operations(binary, 'series_number')
        filtered_series_number = self.filter_contours(series_number_contours, 'series_number', 
                                                    image_width, image_height)
        
        # Обработка каждой области
        owner_fields, birth_place_parts, owner_viz = self.process_owner_information_contours(
            filtered_owner_info, image_width, image_height, binary, padding
        )
        
        number_fields, number_viz = self.process_passport_number_contours(
            filtered_series_number, binary, padding
        )
        
        # Объединение визуализации
        all_visualization = owner_viz + number_viz
        
        # Создание результирующего изображения
        result_image = self.draw_visualization(image, all_visualization)
        
        # Обработка места рождения
        birth_place_field = self.process_birth_place_data(birth_place_parts)
        
        # Формирование итоговой структуры данных
        passport_data = {
            "image_info": {
                "width": image_width,
                "height": image_height
            },
            "fields": owner_fields + number_fields
        }
        
        if birth_place_field:
            passport_data["fields"].append(birth_place_field)
        
        return result_image, passport_data 