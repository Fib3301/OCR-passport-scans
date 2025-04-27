"""
Конфигурационные параметры для обработки изображений
"""

DEFAULT_CONFIG = {
    'target_width': 800,  # Целевая ширина изображения
    'gamma': 1.2,  # Коэффициент гамма-коррекции
    'clahe_clip_limit': 2.0,  # Предел для CLAHE
    'clahe_grid_size': (8, 8),  # Размер сетки для CLAHE
    'gaussian_kernel_size': (5, 5),  # Размер ядра для размытия Гаусса
    'ocr_config': '--oem 3 --psm 6 -l rus+eng',  # Конфигурация OCR
}

# Параметры для морфологических операций
MORPHOLOGY_CONFIG = {
    'owner_info': {
        'close_kernel_size': (30, 3),
        'open_kernel_size': (11, 11)
    }
}

# Параметры фильтрации контуров
CONTOUR_FILTER_CONFIG = {
    'owner_info': {
        'x_min_ratio': 0.35,
        'x_max_ratio': 0.9
    }
} 