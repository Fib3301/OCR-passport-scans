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
    },
    'series_number': {
        'close_kernel_size': (5, 70),
        'open_kernel_size': (11, 11)
    },
    'photo': {
        'close_kernel_size': (5, 50),
        'open_kernel_size': (15, 15)
    }
}

# Параметры фильтрации контуров
CONTOUR_FILTER_CONFIG = {
    'owner_info': {
        'x_min_ratio': 0.35,
        'x_max_ratio': 0.9
    },
    'series_number': {
        'x_min_ratio': 0.9,
        'min_height_width_ratio': 5.0
    },
    'photo': {
        'height_min_ratio': 0.2,
        'height_max_ratio': 0.9,
        'min_area_ratio': 0.05,
        'x_max_ratio': 0.25
    }
} 