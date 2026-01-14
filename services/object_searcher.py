"""
MOD 4: Nesne Arama Servisi
Belirli bir nesneyi görüntüde arar ve konumunu bildirir
"""

# Türkçe nesne isimleri (kendi içinde tanımlı)
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


def search_object(obstacles, target, frame_width, frame_height, report_not_found=True):
    """
    Belirli bir nesneyi ara
    obstacles: [(x1, y1, x2, y2, class_name, confidence, distance), ...]
    target: Aranacak nesne (Türkçe veya İngilizce)
    report_not_found: Bulunamadığında mesaj döndür
    Returns: Bulunan nesnenin konumu veya bulunamadı mesajı
    """
    if not target:
        return None
    
    target = target.lower().strip()
    
    # Türkçe -> İngilizce eşleştirme
    target_en = None
    target_tr = target  # Türkçe ismi sakla
    for en, tr in TURKISH_NAMES.items():
        if target in tr.lower() or target == en.lower():
            target_en = en
            target_tr = tr
            break
    
    # Eğer obstacles boşsa veya None ise
    if not obstacles:
        if report_not_found:
            return f"Burada {target_tr} yok"
        return None
    
    # Nesneyi bul
    for item in obstacles:
        x1, y1, x2, y2 = item[:4]
        class_name = item[4] if len(item) > 4 else ""
        confidence = item[5] if len(item) > 5 else 0
        distance = item[6] if len(item) > 6 else None
        
        # Eşleşme kontrolü
        matched = False
        if target_en and class_name.lower() == target_en:
            matched = True
        elif target in class_name.lower():
            matched = True
        elif target in get_turkish_name(class_name).lower():
            matched = True
        
        if not matched:
            continue
        
        # Merkez noktası
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Bölge belirle (yatay)
        left_end = frame_width // 3
        right_start = 2 * frame_width // 3
        
        if cx < left_end:
            h_position = "solda"
        elif cx > right_start:
            h_position = "sağda"
        else:
            h_position = "önünüzde"
        
        # Türkçe isim
        tr_name = get_turkish_name(class_name)
        
        # Mesafe hesapla (eğer yoksa bbox'tan tahmin et)
        if distance is None:
            # Basit mesafe tahmini: nesne büyüklüğüne göre
            bbox_height = y2 - y1
            # Daha büyük nesne = daha yakın
            if bbox_height > frame_height * 0.6:
                distance = 0.5  # Çok yakın
            elif bbox_height > frame_height * 0.4:
                distance = 1.0
            elif bbox_height > frame_height * 0.25:
                distance = 2.0
            elif bbox_height > frame_height * 0.15:
                distance = 3.0
            else:
                distance = 4.0  # Uzak
        
        # Mesafe açıklaması
        if distance < 1.0:
            distance_text = "çok yakın"
        elif distance < 2.0:
            distance_text = f"yaklaşık {distance:.1f} metre"
        else:
            distance_text = f"yaklaşık {distance:.0f} metre"
        
        return f"{tr_name} {h_position}, {distance_text} uzaklıkta"
    
    # Nesne bulunamadı
    if report_not_found:
        return f"Burada {target_tr} yok"
    return None


def get_available_objects(obstacles):
    """Görüntüdeki tüm nesnelerin Türkçe listesini döndür"""
    if not obstacles:
        return []
    
    objects = []
    for item in obstacles:
        class_name = item[4] if len(item) > 4 else "nesne"
        tr_name = get_turkish_name(class_name)
        if tr_name not in objects:
            objects.append(tr_name)
    
    return objects


# Test
if __name__ == '__main__':
    # Test verisi
    test_obstacles = [
        (100, 100, 200, 300, 'person', 0.9, 2.5),
        (400, 150, 500, 350, 'chair', 0.85, 1.8),
        (600, 200, 700, 400, 'bottle', 0.75, 3.2)
    ]
    
    # İnsan ara
    result = search_object(test_obstacles, "insan", 800, 600)
    print(f"İnsan: {result}")
    
    # Sandalye ara
    result = search_object(test_obstacles, "sandalye", 800, 600)
    print(f"Sandalye: {result}")
    
    # Şişe ara
    result = search_object(test_obstacles, "bottle", 800, 600)
    print(f"Şişe: {result}")
    
    # Mevcut nesneler
    available = get_available_objects(test_obstacles)
    print(f"Mevcut nesneler: {available}")
