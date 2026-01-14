"""
MOD 3: Nesne Tanımlama Servisi
Görüntüdeki nesneleri konumları ve mesafeleriyle tanımlar
"""

# Türkçe nesne isimleri
TURKISH_NAMES = {
    'person': 'insan',
    'bicycle': 'bisiklet',
    'car': 'araba',
    'motorcycle': 'motosiklet',
    'airplane': 'uçak',
    'bus': 'otobüs',
    'train': 'tren',
    'truck': 'kamyon',
    'boat': 'tekne',
    'traffic light': 'trafik lambası',
    'fire hydrant': 'yangın musluğu',
    'stop sign': 'dur işareti',
    'parking meter': 'park sayacı',
    'bench': 'bank',
    'bird': 'kuş',
    'cat': 'kedi',
    'dog': 'köpek',
    'horse': 'at',
    'sheep': 'koyun',
    'cow': 'inek',
    'elephant': 'fil',
    'bear': 'ayı',
    'zebra': 'zebra',
    'giraffe': 'zürafa',
    'backpack': 'sırt çantası',
    'umbrella': 'şemsiye',
    'handbag': 'el çantası',
    'tie': 'kravat',
    'suitcase': 'bavul',
    'frisbee': 'frizbi',
    'skis': 'kayak',
    'snowboard': 'snowboard',
    'sports ball': 'top',
    'kite': 'uçurtma',
    'baseball bat': 'beyzbol sopası',
    'baseball glove': 'beyzbol eldiveni',
    'skateboard': 'kaykay',
    'surfboard': 'sörf tahtası',
    'tennis racket': 'tenis raketi',
    'bottle': 'şişe',
    'wine glass': 'şarap bardağı',
    'cup': 'bardak',
    'fork': 'çatal',
    'knife': 'bıçak',
    'spoon': 'kaşık',
    'bowl': 'kase',
    'banana': 'muz',
    'apple': 'elma',
    'sandwich': 'sandviç',
    'orange': 'portakal',
    'broccoli': 'brokoli',
    'carrot': 'havuç',
    'hot dog': 'sosisli',
    'pizza': 'pizza',
    'donut': 'donut',
    'cake': 'pasta',
    'chair': 'sandalye',
    'couch': 'kanepe',
    'potted plant': 'saksı bitkisi',
    'bed': 'yatak',
    'dining table': 'yemek masası',
    'toilet': 'tuvalet',
    'tv': 'televizyon',
    'laptop': 'dizüstü bilgisayar',
    'mouse': 'fare',
    'remote': 'uzaktan kumanda',
    'keyboard': 'klavye',
    'cell phone': 'cep telefonu',
    'microwave': 'mikrodalga',
    'oven': 'fırın',
    'toaster': 'tost makinesi',
    'sink': 'lavabo',
    'refrigerator': 'buzdolabı',
    'book': 'kitap',
    'clock': 'saat',
    'vase': 'vazo',
    'scissors': 'makas',
    'teddy bear': 'oyuncak ayı',
    'hair drier': 'saç kurutma',
    'toothbrush': 'diş fırçası'
}


def get_turkish_name(english_name):
    """İngilizce nesne ismini Türkçe'ye çevir"""
    return TURKISH_NAMES.get(english_name.lower(), english_name)


def describe_objects(obstacles, frame_width, frame_height):
    """
    Görüntüdeki nesneleri konumları ve mesafeleriyle tanımla
    obstacles: [(x1, y1, x2, y2, class_name, confidence, distance), ...]
    Returns: Türkçe açıklama string
    """
    if not obstacles:
        return "Görüş alanında nesne yok"
    
    left_objects = []
    center_objects = []
    right_objects = []
    
    left_end = frame_width // 3
    right_start = 2 * frame_width // 3
    
    for item in obstacles:
        x1, y1, x2, y2 = item[:4]
        class_name = item[4] if len(item) > 4 else "nesne"
        distance = item[6] if len(item) > 6 else None
        
        # Merkez noktası
        cx = (x1 + x2) // 2
        
        # Türkçe isim
        tr_name = get_turkish_name(class_name)
        
        # Mesafe bilgisi
        if distance:
            obj_info = f"{tr_name} {distance:.1f} metre"
        else:
            obj_info = tr_name
        
        # Bölgeye göre ayır
        if cx < left_end:
            left_objects.append(obj_info)
        elif cx > right_start:
            right_objects.append(obj_info)
        else:
            center_objects.append(obj_info)
    
    # Açıklama oluştur
    parts = []
    
    if center_objects:
        parts.append(f"Önde: {', '.join(center_objects)}")
    if left_objects:
        parts.append(f"Solda: {', '.join(left_objects)}")
    if right_objects:
        parts.append(f"Sağda: {', '.join(right_objects)}")
    
    if parts:
        return ". ".join(parts)
    return "Görüş alanında nesne yok"


# Test
if __name__ == '__main__':
    # Test verisi
    test_obstacles = [
        (100, 100, 200, 300, 'person', 0.9, 2.5),
        (400, 150, 500, 350, 'chair', 0.85, 1.8),
        (600, 200, 700, 400, 'bottle', 0.75, 3.2)
    ]
    
    result = describe_objects(test_obstacles, 800, 600)
    print(f"Sonuç: {result}")
