"""
Основной модуль для запуска обработки изображений паспортов
"""

import argparse
from pathlib import Path
import numpy as np
from utils import resize_image, process_image
from contour_processor import ContourProcessor
import cv2
import json
from config import DEFAULT_CONFIG

def process_directory(input_dir: str, output_dir: str, config: dict = None) -> None:
    """
    Обработка всех изображений в директории
    
    Args:
        input_dir: Путь к входной директории
        output_dir: Путь к выходной директории
        config: Конфигурация параметров обработки
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if config is None:
        config = DEFAULT_CONFIG
    image_extensions = ('.jpg', '.png')
    
    # Получаем список всех изображений в директории
    images = [img for img in input_path.glob('*') 
             if img.suffix.lower() in image_extensions]
    
    if not images:
        print(f"Изображения не найдены в директории {input_dir}")
        return
    
    # Путь к директории с идеальными данными
    ground_truth_dir = Path("correct_passport_data")
    
    contour_processor = ContourProcessor(config)
    
    # Обработка каждого изображения
    for image_path in images:
        try:
            print("\n" + "="*50)
            print(f"ОБРАБОТКА ФАЙЛА: {image_path.name}")
            print("="*50)
            
            # Загрузка и изменение размера изображения
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Не удалось прочитать изображение: {image_path}")
            image = resize_image(image, config['target_width'])
            
            # Обработка изображения
            binary = process_image(image_path, config)
            
            # Путь к идеальным данным для текущего изображения
            ground_truth_path = ground_truth_dir / f"{image_path.stem}_passport.json"
            
            # Обработка контуров и рисование результатов
            result_image, passport_data = contour_processor.process_contours(
                image, 
                binary, 
                ground_truth_path=ground_truth_path if ground_truth_path.exists() else None
            )

            # Сохранение результата
            output_file = output_path / f"{image_path.stem}_contours.jpg"
            cv2.imwrite(str(output_file), result_image)
            
            # Сохранение JSON файла
            json_file = output_path / f"{image_path.stem}_passport.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(passport_data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"Ошибка при обработке {image_path.name}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description='Обработка изображений паспортов')
    parser.add_argument('--input', '-i', default='img',
                      help='Путь к входной директории (по умолчанию: img)')
    parser.add_argument('--output', '-o', default='output',
                      help='Путь к выходной директории (по умолчанию: output)')
    parser.add_argument('--config', '-c', help='Путь к файлу конфигурации (опционально)')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации, если указана
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    process_directory(args.input, args.output, config)

if __name__ == "__main__":
    main() 