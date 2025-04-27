"""
Модуль для расчета метрик качества распознавания
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from difflib import SequenceMatcher

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Вычисление IoU (Intersection over Union) между двумя прямоугольниками
    
    Args:
        box1: [x, y, width, height] первого прямоугольника
        box2: [x, y, width, height] второго прямоугольника
        
    Returns:
        float: Значение IoU
    """
    # Получаем координаты углов прямоугольников
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Вычисляем координаты пересечения
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    # Если нет пересечения, возвращаем 0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Вычисляем площади
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    # Вычисляем IoU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    
    return iou

def calculate_character_accuracy(pred_text: str, gt_text: str) -> float:
    """
    Вычисление точности распознавания символов (Character Accuracy)
    
    Args:
        pred_text: Предсказанный текст
        gt_text: Истинный текст
        
    Returns:
        float: Значение Character Accuracy
    """
    if not gt_text:
        return 0.0 if pred_text else 1.0
    
    # Используем SequenceMatcher для сравнения строк
    matcher = SequenceMatcher(None, pred_text, gt_text)
    
    # Получаем количество совпадающих символов
    matches = sum(size for _, _, size in matcher.get_matching_blocks())
    
    # Вычисляем точность
    accuracy = matches / len(gt_text)
    
    return accuracy

def evaluate_ocr_quality(pred_data: Dict[str, Any], gt_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Оценка качества OCR для всех полей
    
    Args:
        pred_data: Предсказанные данные
        gt_data: Истинные данные
        
    Returns:
        Dict[str, float]: Словарь с метриками для каждого типа поля
    """
    metrics = {}
    
    # Создаем словарь с истинными значениями для быстрого доступа
    gt_fields = {field['type']: field for field in gt_data['fields']}
    
    # Оцениваем каждое предсказанное поле
    for pred_field in pred_data['fields']:
        field_type = pred_field['type']
        
        # Пропускаем поля без текста (например, фото)
        if 'value' not in pred_field:
            continue
        
        # Если есть соответствующее поле в истинных данных
        if field_type in gt_fields:
            gt_field = gt_fields[field_type]
            
            # Вычисляем IoU для области
            iou = calculate_iou(pred_field['bbox'], gt_field['bbox'])
            
            # Вычисляем точность распознавания текста
            char_accuracy = calculate_character_accuracy(
                pred_field['value'], 
                gt_field['value']
            )
            
            # Сохраняем метрики
            metrics[field_type] = {
                'iou': iou,
                'char_accuracy': char_accuracy
            }
    
    return metrics

def calculate_average_metrics(metrics_list: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Вычисление средних метрик по всем изображениям
    
    Args:
        metrics_list: Список метрик для каждого изображения
        
    Returns:
        Dict[str, Dict[str, float]]: Средние метрики для каждого типа поля
    """
    if not metrics_list:
        return {}
    
    # Собираем все типы полей
    field_types = set()
    for metrics in metrics_list:
        field_types.update(metrics.keys())
    
    # Вычисляем средние значения для каждого типа поля
    avg_metrics = {}
    for field_type in field_types:
        ious = [m[field_type]['iou'] for m in metrics_list if field_type in m]
        accuracies = [m[field_type]['char_accuracy'] for m in metrics_list if field_type in m]
        
        avg_metrics[field_type] = {
            'iou': np.mean(ious) if ious else 0.0,
            'char_accuracy': np.mean(accuracies) if accuracies else 0.0
        }
    
    return avg_metrics 